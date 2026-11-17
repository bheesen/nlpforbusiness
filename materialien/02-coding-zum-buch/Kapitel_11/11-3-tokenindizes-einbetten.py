# Tokenindizes mit einer Embedding-Schicht abbilden
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import baue_vokabular, kodiere, setze_seed
import torch

setze_seed()
woerter, index = baue_vokabular()
ids = torch.tensor([kodiere("pumpe verliert öl", index)])
schicht = torch.nn.Embedding(
    num_embeddings=len(woerter),
    embedding_dim=5,
    padding_idx=index["<pad>"],
)
vektoren = schicht(ids)
# Summe nur bilden, um den Backward Pass zu demonstrieren.
summe = vektoren.sum()
summe.backward()
print("Tokenindizes:", ids.tolist()[0])
print("Ausgabeform: ", tuple(vektoren.shape))
print("Vektor Pumpe:", vektoren[0, 0].detach().round(decimals=3))
print("Gradient PAD:", schicht.weight.grad[index["<pad>"]].tolist()) 
