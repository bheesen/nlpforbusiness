# Ein mehrschichtiges Netz auf Bag-of-Words-Merkmalen
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import KLASSEN, bow_daten, genauigkeit, setze_seed, trainiere
import torch

setze_seed()
woerter, _, x, y = bow_daten()
modell = torch.nn.Sequential(
    torch.nn.Linear(len(woerter), 12),
    torch.nn.ReLU(),
    torch.nn.Linear(12, len(KLASSEN)),
)
for epoche, loss in trainiere(modell, x, y):
    print(f"Epoche {epoche:>3}: Verlust={loss:.4f}")
with torch.no_grad():
    logits = modell(x)
print(f"Trainingsgenauigkeit: {genauigkeit(logits, y):.3f}")
for i in [0, 7, 14]:
    print(f"{i:>2}: {KLASSEN[logits[i].argmax()]} ")
