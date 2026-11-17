# Eine Paddingmaske schließt künstliche Positionen aus.
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
x = torch.randn(2, 5, 8)
maske = torch.tensor([
    [False, False, False, False, False],
    [False, False, False, True, True],
])
modul = torch.nn.MultiheadAttention(8, 2, batch_first=True)
_, gewichte = modul(x, x, x, key_padding_mask=maske)
zeile = gewichte[1, 0]

print("Maske zweiter Satz:", maske[1].tolist())
werte = [round(float(v), 3) for v in zeile.detach()]
print("Gewichte Anfrage 0:", werte)
print("Summe auf Padding:", round(float(zeile[3:].sum().detach()), 6))
print("Gesamtsumme:      ", round(float(zeile.sum().detach()), 3))
