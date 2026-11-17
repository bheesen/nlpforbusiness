# Tensoren, Verlustfunktion und automatische Differentiation
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
x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
y = torch.tensor([0, 1])
schicht = torch.nn.Linear(2, 2)
logits = schicht(x)
loss = torch.nn.functional.cross_entropy(logits, y)
loss.backward()
print("Eingabeform:", tuple(x.shape))
print("Logitform: ", tuple(logits.shape))
print(f"Verlust:    {loss.item():.4f}")
print(f"Gradient W: {schicht.weight.grad.norm().item():.4f}")
print(f"Gradient b: {schicht.bias.grad.norm().item():.4f}") 
