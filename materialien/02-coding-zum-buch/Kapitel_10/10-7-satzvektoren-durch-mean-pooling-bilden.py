# Satzvektoren durch Mean Pooling vergleichen
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import kosinus, satzvektor, svd_embedding

woerter, vektoren = svd_embedding(dimension=8)
saetze = [
    "pumpe verliert öl",
    "dichtung an pumpe undicht",
    "sensor meldet falsche position",
    "ersatzteile liegen im lager",
]
query = satzvektor(saetze[0], woerter, vektoren)
for satz in saetze[1:]:
    wert = kosinus(query, satzvektor(satz, woerter, vektoren))
    print(f"{wert:.3f}  {satz}")
