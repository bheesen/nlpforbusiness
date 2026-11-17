# Positionskodierungen unterscheiden gleiche Token an verschiedenen Orten
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import positionskodierung, setze_seed
import torch
print(positionskodierung)
setze_seed()
wort = torch.randn(6)
ohne = torch.stack([wort, wort])
mit = ohne + positionskodierung(2, 6)
cos = torch.nn.functional.cosine_similarity
a = float(cos(ohne[0], ohne[1], dim=0))
print("Ähnlichkeit ohne Position:", round(a, 3))
print("Ähnlichkeit mit Position: ", round(float(cos(mit[0], mit[1], dim=0)), 3))
print("Position 0:", mit[0].round(decimals=3).tolist())
print("Position 1:", mit[1].round(decimals=3).tolist()) 
