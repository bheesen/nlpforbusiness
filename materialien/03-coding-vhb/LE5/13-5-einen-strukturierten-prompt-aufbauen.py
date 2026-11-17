import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CUDA-GPU verwenden, sonst CPU nutzen
geraet = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Rechengerät:", geraet)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

modellname = (
    "Qwen/Qwen3-1.7B"
    if geraet.type == "cuda" else "Qwen/Qwen3-0.6B"
)
dtype = torch.float16 if geraet.type == "cuda" else torch.float32
print("Modell:", modellname)

tokenizer = AutoTokenizer.from_pretrained(modellname)

modell = AutoModelForCausalLM.from_pretrained(
    modellname,
    dtype=dtype,
).to(geraet)

modell.eval()

bericht = "Pumpe 4 verliert Kühlwasser. Priorität hoch."

prompt_1 = f"Extrahiere die Informationen:\n{bericht}"
prompt_2 = f"""Extrahiere nur Angaben aus dem Bericht.
Bericht: {bericht}
Gib nur gültiges JSON mit anlage, problem und prioritaet zurück.
Fehlende Angaben erhalten den Wert null."""

# Qwen-Antwort erzeugen
def qwen3_antwort(prompt):
    chat = [{"role": "user", "content": prompt}]
    daten = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(geraet)
    with torch.inference_mode():
        tokens = modell.generate(
            **daten,
            max_new_tokens=80,
            do_sample=False,
        )
    start = daten["input_ids"].shape[-1]
    return tokenizer.decode(
        tokens[0][start:],
        skip_special_tokens=True,
    ).strip()

# JSON aus der LLM-Antwort extrahieren und prüfen
def json_lesen(text):
    treffer = re.search(r"\{.*\}", text, re.DOTALL)
    if not treffer:
        return None
    try:
        return json.loads(treffer.group())
    except json.JSONDecodeError:
        return None

antwort_1 = qwen3_antwort(prompt_1)
antwort_2 = qwen3_antwort(prompt_2)
json_daten = json_lesen(antwort_2)
print("Einfacher Prompt:", antwort_1)
print("Strukturierter Prompt:", antwort_2)
print("JSON gültig:", json_daten is not None)
if json_daten is not None:
    print("Geparstes JSON:", json_daten)