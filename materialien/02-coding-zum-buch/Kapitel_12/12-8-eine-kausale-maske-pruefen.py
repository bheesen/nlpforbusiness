# Eine kausale Maske verhindert den Blick auf zukünftige Token
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import setze_seed
import torch

setze_seed()
tokens = ["pumpe", "verliert", "öl", "<eos>"]
x = torch.randn(1, len(tokens), 8)
maske = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
modul = torch.nn.MultiheadAttention(8, 2, batch_first=True)
_, gewichte = modul(x, x, x, attn_mask=maske)
print("Kausale Maske:")
for zeile in maske.int().tolist():
    print(" ", zeile)
print("Gewichte bei 'verliert':")
print(" ", gewichte[0, 1].round(decimals=3).tolist())
zukunft = float(gewichte[0, 1, 2:].sum().detach())
print("Gewicht auf Zukunft:", round(zukunft, 6)) 
