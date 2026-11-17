# Vortrainiertes Instruct-Modell fasst einen Wartungsbericht zusammen.
#%% Verzeichnisse definieren
from pathlib import Path
import sys
BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS
DOKUMENTE = EigeneDaten / "Organisation"
#%% Pakete importieren
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELL = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(MODELL)
modell = AutoModelForCausalLM.from_pretrained(
    MODELL,
    dtype="auto",
    device_map="auto",)
bericht = (DOKUMENTE / "09_stoerungsbericht_zp4.txt").read_text(
    encoding="utf-8")
# Das Modell soll nur Informationen aus dem Bericht verdichten.
nachrichten = [{"role": "system",
                "content": (
                "Fasse technische Berichte auf Deutsch zusammen. "
                "Erfinde keine Angaben."),},
               {"role": "user",
                "content": (
                "Gib Anlage, Alarm, Messwert, Status und Priorität "
                f"als kurze Liste aus.\n\nBERICHT:\n{bericht}"), },]
text = tokenizer.apply_chat_template(
    nachrichten,
    tokenize=False,
    add_generation_prompt=True,)
eingabe = tokenizer(text, return_tensors="pt").to(modell.device)
ausgabe = modell.generate(
    **eingabe,
    max_new_tokens=160,
    do_sample=False,)
neu = ausgabe[0][eingabe["input_ids"].shape[1]:]
print(tokenizer.decode(neu, skip_special_tokens=True))
