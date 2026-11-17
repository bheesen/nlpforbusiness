# Eine vollständige Transformer-Pipeline klassifiziert neue Meldungen
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import (
    KLASSEN,
    Klassifikator,
    kodiere,
    sequenzdaten,
    setze_seed,
    trainiere,
)
import torch

setze_seed()
woerter, index, x, y = sequenzdaten()
modell = Klassifikator(len(woerter))
trainiere(modell, x, y, epochen=90)
neu = [
    "pumpe verliert kühlwasser",
    "motorlager rattert laut",
    "sensor liefert falsche werte",
    "anlage meldet problem",
]
ids = torch.tensor([kodiere(text, index) for text in neu])
with torch.no_grad():
    probs = modell(ids).softmax(1)
for text, zeile, verteilung in zip(neu, ids, probs):
    wert, klasse = verteilung.max(0)
    unbekannt = int(zeile.eq(index["<unk>"]).sum())
    sicher = float(wert) >= 0.75 and unbekannt < 2
    status = "automatisch" if sicher else "manuell prüfen"
    print(text)
    print(f"  Klasse: {KLASSEN[int(klasse)]}")
    print(f"  Sicherheit: {float(wert):.3f} | {status}")
