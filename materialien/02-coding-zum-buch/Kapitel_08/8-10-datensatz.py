# Einen gelabelten Textdatensatz untersuchen
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from collections import Counter
from nlp_eda import WARTUNGSDATEN_TEXTE, WARTUNGSDATEN_LABELS

texte, labels = WARTUNGSDATEN_TEXTE, WARTUNGSDATEN_LABELS
verteilung = Counter(labels)
print(f"Dokumente: {len(texte)}")
print(f"Klassen:   {len(verteilung)}")
print("Klasse       Anzahl  Beispiel")
print("-" * 66)
for klasse in sorted(verteilung):
    index = labels.index(klasse)
    print(f"{klasse:<12} {verteilung[klasse]:>6}  {texte[index]}")
