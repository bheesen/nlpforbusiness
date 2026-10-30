# Eine Wort-Wort-Kookkurrenzmatrix aufbauen
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import SAETZE, kookkurrenz

print("Texte:")
for nr, doc in enumerate(SAETZE, start=1):
    print(f"Text-{nr}: {doc}")
woerter, matrix = kookkurrenz(fenster=1)
zeilen = ["pumpe", "motor", "lager", "dichtung", "sensor"]
spalten_w = ["vibriert", "defekt", "druck", "meldet", "im"]
spalten = [woerter.index(wort) for wort in spalten_w]
print("\nWort       " + " ".join(f"{w:>9}" for w in spalten_w))
print("-" * 60)
for wort in zeilen:
    zeile = matrix[woerter.index(wort), spalten]
    werte = " ".join(f"{int(wert):>9}" for wert in zeile)
    print(f"{wort:<10} {werte}")
