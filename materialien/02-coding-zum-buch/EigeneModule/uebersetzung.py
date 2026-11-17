# Neuronale Übersetzung mit lokal gespeicherten OPUS-MT-Modellen.
from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pfade import modellordner


@lru_cache(maxsize=2)
def lade_modell(richtung):
    # Jede Übersetzungsrichtung beim ersten Aufruf einmal laden.
    if richtung not in ("de-en", "en-de"):
        raise ValueError("Richtung muss de-en oder en-de sein")
    pfad = modellordner() / richtung
    tok = AutoTokenizer.from_pretrained(pfad, local_files_only=True)
    modell = AutoModelForSeq2SeqLM.from_pretrained(
        pfad, local_files_only=True)
    modell.eval()
    return tok, modell


def uebersetzen(text, richtung="de-en"):
    # Leere Eingaben zurückweisen und neuronale Inferenz ausführen.
    if not text.strip():
        raise ValueError("Bitte Text eingeben")
    tok, modell = lade_modell(richtung)
    daten = tok(text, return_tensors="pt", truncation=True)
    # Übersetzung mit maximal 128 Tokens erzeugen
    with torch.inference_mode():
        tokens = modell.generate(**daten, max_length=128)
    return tok.decode(tokens[0], skip_special_tokens=True)
