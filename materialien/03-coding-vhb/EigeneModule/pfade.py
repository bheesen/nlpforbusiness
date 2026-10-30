#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"


def modellordner():
    # In der EXE liegen Modelle neben der ausführbaren Datei.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "modelle"
    return EigeneDaten / "modelle"


def ragordner():
    # In der EXE liegen der Index und die Metadaten neben der EXE.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "rag"
    return EigenerOutput
