# Gradient Clipping begrenzt sehr große Gradienten
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
rnn = torch.nn.RNN(5, 8, batch_first=True)
x = torch.randn(4, 25, 5) * 8
ziel = torch.zeros(4, 25, 8)
ausgabe, _ = rnn(x)
loss = ((ausgabe - ziel) ** 2).sum() * 20
loss.backward()
parameter = [p for p in rnn.parameters() if p.grad is not None]
vorher = torch.sqrt(sum((p.grad ** 2).sum() for p in parameter))
torch.nn.utils.clip_grad_norm_(parameter, max_norm=1.0)
nachher = torch.sqrt(sum((p.grad ** 2).sum() for p in parameter))
print(f"Verlust: {loss.item():.2f}")
print(f"Gradientennorm vorher: {vorher.item():.3f}")
print(f"Gradientennorm nachher: {nachher.item():.3f}") 
