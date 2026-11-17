# Ein Transformer-Encoder klassifiziert Wartungsberichte über CLS
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
    genauigkeit,
    sequenzdaten,
    setze_seed,
    trainiere,
)
import torch

setze_seed()
woerter, _, x, y = sequenzdaten()
modell = Klassifikator(len(woerter))
for epoche, loss in trainiere(modell, x, y, epochen=80):
    print(f"Epoche {epoche:>2}: Verlust={loss:.4f}")
with torch.no_grad():
    logits = modell(x)
print("Trainingsgenauigkeit:", round(genauigkeit(logits, y), 3))
for i in [0, 6, 12]:
    print(f"{i:>2}: {KLASSEN[int(logits[i].argmax())]}")
