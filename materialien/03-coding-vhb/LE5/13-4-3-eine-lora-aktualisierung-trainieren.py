#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Module laden
import json
import re
import torch
from datasets import Dataset, disable_progress_bar
from nlp_eda import wartungsmeldungen
from nlp_eda import wartungsmeldungen_testdaten
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

disable_progress_bar()
# CUDA-GPU verwenden, sonst CPU nutzen
geraet = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Rechengerät:", geraet)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

MODELL = "Qwen/Qwen3-0.6B"

system = (
    "Ordne die Wartungsmeldung einer Kategorie und Priorität zu. "
    "Kategorien: leckage, sensorik, mechanik. "
    "Prioritäten: hoch, mittel, niedrig. "
    "Antworte nur mit einem JSON-Objekt."
)

#%% Trainingsdaten aufbauen
rohdaten = []
for text, kategorie, prioritaet in wartungsmeldungen:
    ziel = {"kategorie": kategorie, "prioritaet": prioritaet}
    rohdaten.append({"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
        {"role": "assistant", "content":
         json.dumps(ziel, ensure_ascii=False)},
    ]})

#%% Tokenizer und Modell laden
tokenizer = AutoTokenizer.from_pretrained(MODELL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

basis = AutoModelForCausalLM.from_pretrained(
    MODELL,
    dtype=torch.float32,
).to(geraet)
basis.config.use_cache = False

#%% LoRA-Adapter ergänzen
lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
modell = get_peft_model(basis, lora)
gesamt = sum(p.numel() for p in modell.parameters())
trainierbar = sum(
    p.numel() for p in modell.parameters() if p.requires_grad
)

#%% Modellantwort bereinigen
def bereinigen(text):
    text = text.replace("�", " ")
    text = " ".join(text.split())
    treffer = re.search(r"\{.*?\}", text)
    return treffer.group() if treffer else text[:160]

#%% Wartungsmeldung klassifizieren
def klassifizieren(text):
    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    eingabe = tokenizer(prompt, return_tensors="pt").to(geraet)
    modell.eval()
    with torch.no_grad():
        ids = modell.generate(
            **eingabe,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    start = eingabe["input_ids"].shape[1]
    text = tokenizer.decode(
        ids[0][start:],
        skip_special_tokens=True,
    )
    return bereinigen(text)

#%% Testdaten auswerten
def testen():
    korrekt = 0
    for text, soll_kat, soll_prio in wartungsmeldungen_testdaten:
        antwort = klassifizieren(text)
        try:
            daten = json.loads(antwort)
        except json.JSONDecodeError:
            continue
        kat = daten.get("kategorie", daten.get("category", ""))
        prio = daten.get("prioritaet", daten.get("priority", ""))
        korrekt += int(kat == soll_kat and prio == soll_prio)
    return korrekt

#%% Test vor LoRA
korrekt_vorher = testen()

#%% Nur Assistant-Antwort als Lernziel verwenden
def vorbereiten(datensatz):
    nachrichten = datensatz["messages"]
    prompt = tokenizer.apply_chat_template(
        nachrichten[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    gesamt = tokenizer.apply_chat_template(
        nachrichten,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]
    token = tokenizer(
        gesamt,
        add_special_tokens=False,
        truncation=True,
        max_length=192,
    )
    labels = token["input_ids"].copy()
    grenze = min(len(prompt_ids), len(labels))
    labels[:grenze] = [-100] * grenze
    token["labels"] = labels
    return token

#%% Trainingsdaten tokenisieren
daten = Dataset.from_list(rohdaten)
daten = daten.map(
    vorbereiten,
    remove_columns=daten.column_names,
)
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
)
modell.train()

epochen = 2 if geraet.type == "cuda" else 0.2

#%% LoRA trainieren
argumente = TrainingArguments(
    output_dir=str(EigenerOutput / "qwen3-lora"),
    num_train_epochs=epochen,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    save_strategy="no",
    logging_strategy="no",
    disable_tqdm=True,
    report_to="none",
)
trainer = Trainer(
    model=modell,
    args=argumente,
    train_dataset=daten,
    data_collator=collator,
)
trainer.train()

#%% Adapter speichern und erneut testen
adapterpfad = EigenerOutput / "qwen3-lora-adapter"
modell.save_pretrained(adapterpfad)
tokenizer.save_pretrained(adapterpfad)
modell.config.use_cache = True
korrekt_nachher = testen()

print("Gesamtparameter:", gesamt)
print("Trainierbare Parameter:", trainierbar)
print("Anteil trainierbar:", f"{100 * trainierbar / gesamt:.3f} %")
print("Korrekt vor LoRA:", korrekt_vorher, "von 9")
print("Korrekt nach LoRA:", korrekt_nachher, "von 9")
print("Adapter gespeichert:", adapterpfad)