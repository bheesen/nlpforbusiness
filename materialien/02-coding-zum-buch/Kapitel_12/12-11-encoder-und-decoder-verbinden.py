# Ein Encoder-Decoder-Transformer verarbeitet Quell- und Zielsequenzen
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
modell = torch.nn.Transformer(
    d_model=16,
    nhead=4,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=32,
    dropout=0.0,
    batch_first=True,
)
quelle = torch.randn(2, 5, 16)
ziel = torch.randn(2, 4, 16)
maske = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
ausgabe = modell(quelle, ziel, tgt_mask=maske)
print("Quellform:  ", tuple(quelle.shape))
print("Zielform:   ", tuple(ziel.shape))
print("Ausgabeform:", tuple(ausgabe.shape))
print("Zielmaske:")
for zeile in maske.int().tolist():
    print(" ", zeile)
