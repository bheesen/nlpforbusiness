# Dropout begrenzt die Abhängigkeit von einzelnen Aktivierungen
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import bow_daten, setze_seed
import torch

def trainiere(dropout):
    setze_seed()
    _, _, x, y = bow_daten()
    modell = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 24),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(24, 3),
    )
    opt = torch.optim.Adam(modell.parameters(), lr=0.03)
    for _ in range(181):
        modell.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(modell(x), y)
        loss.backward()
        opt.step()
    modell.train()
    a = modell(x[:1])
    b = modell(x[:1])
    modell.eval()
    c = modell(x[:1])
    d = modell(x[:1])
    train_diff = float((a - b).abs().max().detach())
    eval_diff = float((c - d).abs().max().detach())
    return train_diff, eval_diff


for rate in [0.0, 0.5]:
    train_diff, eval_diff = trainiere(rate)
    print(f"Dropout={rate:.1f} Train-Differenz={train_diff:.3f}")
    print(f"            Eval-Differenz={eval_diff:.3f}")
