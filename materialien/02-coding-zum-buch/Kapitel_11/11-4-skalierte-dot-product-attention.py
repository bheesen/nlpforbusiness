# Einfache und Multi-Head-Attention für eine kurze Wortfolge.
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
import math
import torch

setze_seed()
tokens = ["sensor", "meldet", "druck", "falsch"]
x = torch.randn(1, len(tokens), 6)

# Eine einzelne skalierte Dot-Product-Attention berechnen.
q = torch.nn.Linear(6, 6, bias=False)(x)
k = torch.nn.Linear(6, 6, bias=False)(x)
v = torch.nn.Linear(6, 6, bias=False)(x)

werte = q @ k.transpose(1, 2) / math.sqrt(q.shape[-1])
gewichte = torch.softmax(werte, dim=-1)
zeile = gewichte[0, 2]

print("Einfache Attention für: druck")
for token, gewicht in zip(tokens, zeile):
    print(f"  {token:<8} {gewicht.item():.3f}")
print("Gewichtssumme:", round(float(zeile.sum().detach()), 3))

# Dieselbe Sequenz mit zwei Attention-Köpfen verarbeiten.
mha = torch.nn.MultiheadAttention(
    embed_dim=6,
    num_heads=2,
    batch_first=True,
)
kontext, kopfgewichte = mha(
    x,
    x,
    x,
    need_weights=True,
    average_attn_weights=False,
)

for nr, kopf in enumerate(kopfgewichte[0, :, 2], start=1):
    print(f"Kopf {nr}:")
    for token, gewicht in zip(tokens, kopf):
        print(f"  {token:<8} {gewicht.item():.3f}")
    print("  Summe:", round(float(kopf.sum().detach()), 3))

print("Multi-Head-Kontextform:", tuple(kontext.shape))
