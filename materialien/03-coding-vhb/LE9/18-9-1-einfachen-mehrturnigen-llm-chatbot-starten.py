
# Ein Instruct-Sprachmodell als mehrturnigen Chatbot verwenden.
#%% Pakete importieren
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#%% Modell und Tokenizer laden
MODELL = "Qwen/Qwen3-4B-Instruct-2507"
print("PyTorch:", torch.__version__)
print("PyTorch-CUDA:", torch.version.cuda)
print("CUDA verfügbar:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Grafikkarte:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(MODELL)
modell = AutoModelForCausalLM.from_pretrained(
    MODELL,
    dtype="auto",
    device_map="auto",
)
modell.eval()

#%% Tatsächliche Gerätebelegung untersuchen
# Jedes Transformer-Layer wird anhand seiner Parameter erfasst.
geraete = {}
for name, parameter in modell.named_parameters():
    if name.startswith("model.layers."):
        nummer = int(name.split(".")[2])
        geraete.setdefault(nummer, set()).add(str(parameter.device))

print("\nModellverteilung")
print("-" * 30)
for geraet in sorted({g for werte in geraete.values() for g in werte}):
    nummern = sorted(i for i, werte in geraete.items() if geraet in werte)
    print(f"{geraet}: {len(nummern)} Transformer-Layer")
    print("Layer-Nummern:", nummern)

print("Embedding:", modell.get_input_embeddings().weight.device)
ausgabeschicht = modell.get_output_embeddings()
if ausgabeschicht is not None:
    print("Ausgabeschicht:", ausgabeschicht.weight.device)

#%% Zwei Chatbots mit verschiedenen System-Prompts vergleichen
fragen = [
    "Was bedeutet VIB-H bei ZP4?",
    "Welches Lager soll ich dann prüfen?",
]

system_prompts = [
    (
        "Chatbot mit strenger Vorgabe",
        "Du bist ein Assistent für Instandhaltung. "
        "Antworte kurz und grammatisch korrekt auf Deutsch. "
        "Erfinde keine organisationsinternen Begriffe oder "
        "Abkürzungen. Wenn dir zu einer Abkürzung keine "
        "verlässliche Information vorliegt, antworte genau: "
        "Dazu liegen mir keine verlässlichen Informationen vor.",
    ),
    (
        "Chatbot mit weniger strenger Vorgabe",
        "Du bist ein Assistent für Instandhaltung. "
        "Antworte kurz auf Deutsch. "
        "Erfinde keine organisationsinternen Abkürzungen.",
    ),
]

# Der Verlauf wird für jeden Chatbot separat aufgebaut.
for titel, system_prompt in system_prompts:
    print(f"\n{titel}")
    print("-" * 30)
    verlauf = [{"role": "system", "content": system_prompt}]

    for frage in fragen:
        verlauf.append({"role": "user", "content": frage})
        # Alle bisherigen Nachrichten in Chat-Tokens umwandeln.
        eingabe = tokenizer.apply_chat_template(
            verlauf,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(modell.device)

        # Inferenz ohne Gradienten und ohne Zufalls-Sampling.
        with torch.inference_mode():
            ausgabe = modell.generate(
                **eingabe,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Nur die neu generierten Tokens dekodieren.
        anzahl = eingabe["input_ids"].shape[1]
        neu = ausgabe[0][anzahl:]
        antwort = tokenizer.decode(
            neu, skip_special_tokens=True
        ).strip()
        print("Frage:", frage)
        print("Antwort:", antwort)
        verlauf.append({"role": "assistant", "content": antwort})

