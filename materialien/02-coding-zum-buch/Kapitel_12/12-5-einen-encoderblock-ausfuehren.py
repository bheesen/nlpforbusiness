# Ein Encoderblock kombiniert Attention und Feedforward-Netz.
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(
    r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP"
)
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import setze_seed
import torch

setze_seed()
schicht = torch.nn.TransformerEncoderLayer(
    d_model=12,
    nhead=3,
    dim_feedforward=24,
    dropout=0.0,
    batch_first=True,
)
x = torch.randn(2, 5, 12)
y = schicht(x)

aenderung = float((y - x).abs().mean().detach())
mittel = float(y[0, 0].mean().detach())
std = float(y[0, 0].std(unbiased=False).detach())
print("Eingabeform:       ", tuple(x.shape))
print("Ausgabeform:       ", tuple(y.shape))
print("Mittlere Änderung: ", round(aenderung, 3))
print("Mittelwert Position:", round(mittel, 6))
print("Std. Position:      ", round(std, 3))
