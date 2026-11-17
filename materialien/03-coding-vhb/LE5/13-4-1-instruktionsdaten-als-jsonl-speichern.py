# Aus Rohtexten mit Qwen Instruction-Tuning-Daten erzeugen
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CUDA-GPU verwenden, sonst CPU nutzen
geraet = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Rechengerät:", geraet)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

MODELL = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(MODELL)
modell = AutoModelForCausalLM.from_pretrained(MODELL).to(geraet)
modell.eval()

berichte = [
    "Pumpe 4 verliert Kühlwasser. Anlage sofort prüfen.",
    "Sensor 8 liefert seit gestern unplausible Messwerte.",
    "Motorlager rattert laut. Austausch dringend empfohlen.",
]

system = (
    "Ordne die Wartungsmeldung einer Kategorie und Priorität zu. "
    "Kategorien: leckage, sensorik, mechanik. "
    "Prioritäten: hoch, mittel, niedrig. "
    "Antworte nur mit einem JSON-Objekt."
)


def analysiere(text):
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
    eingabe = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(geraet)

    with torch.no_grad():
        ids = modell.generate(
            **eingabe,
            max_new_tokens=60,
            do_sample=False,
        )

    start = eingabe["input_ids"].shape[1]
    neu = ids[0][start:]
    return tokenizer.decode(
        neu,
        skip_special_tokens=True,
    ).strip()


def json_lesen(antwort):
    # Erstes JSON-Objekt aus der LLM-Antwort extrahieren
    treffer = re.search(r"\{.*?\}", antwort, re.DOTALL)
    if not treffer:
        raise ValueError(f"Kein JSON gefunden: {antwort!r}")
    return json.loads(treffer.group())


daten = []

for bericht in berichte:
    antwort = analysiere(bericht)
    print("\nBericht:", bericht)
    print("Antwort:", antwort)

    ziel = json_lesen(antwort)

    datensatz = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": bericht},
            {
                "role": "assistant",
                "content": json.dumps(
                    ziel,
                    ensure_ascii=False,
                ),
            },
        ]
    }
    daten.append(datensatz)

with open(
    "instruction_daten.jsonl",
    "w",
    encoding="utf-8",
) as datei:
    for datensatz in daten:
        zeile = json.dumps(
            datensatz,
            ensure_ascii=False,
        )
        datei.write(zeile + "\n")

print("\nErzeugte Trainingsbeispiele:", len(daten))