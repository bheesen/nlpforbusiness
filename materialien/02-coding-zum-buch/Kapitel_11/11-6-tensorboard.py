# Trainingsverlauf mit TensorBoard protokollieren.
#%% Verzeichnisse definieren
from pathlib import Path
import shutil
import sys

BASIS = Path(
    r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP"
)
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import bow_daten, setze_seed
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

#%% Daten reproduzierbar in Training und Validierung teilen
setze_seed()
_, _, x, y = bow_daten()
train_idx = torch.tensor([0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15])
val_idx = torch.tensor([4, 5, 10, 11, 16, 17])
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

#%% Kleines Feedforward-Netz definieren
modell = torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 3),
)
opt = torch.optim.Adam(modell.parameters(), lr=0.03)

# Alte Logs dieses Lehrbeispiels entfernen.
log_dir = EigenerOutput / "tensorboard"
shutil.rmtree(log_dir, ignore_errors=True)
writer = SummaryWriter(log_dir=str(log_dir))

def bewerten(x_daten, y_daten):
    # Loss und Accuracy ohne Gradienten berechnen.
    logits = modell(x_daten)
    loss = F.cross_entropy(logits, y_daten)
    acc = (logits.argmax(1) == y_daten).float().mean()
    return loss.item(), acc.item()

print(
    f"{'Epoche':>6} {'Train-Loss':>10} {'Train-Acc':>9} "
    f"{'Val-Loss':>9} {'Val-Acc':>8}"
)

for epoche in range(101):
    # Einen Optimierungsschritt auf Trainingsdaten ausführen.
    modell.train()
    opt.zero_grad()
    loss = F.cross_entropy(modell(x_train), y_train)
    loss.backward()
    opt.step()
    # Trainings- und Validierungswerte protokollieren.
    modell.eval()
    with torch.no_grad():
        train_loss, train_acc = bewerten(x_train, y_train)
        val_loss, val_acc = bewerten(x_val, y_val)
    writer.add_scalar("Loss/Training", train_loss, epoche)
    writer.add_scalar("Loss/Validierung", val_loss, epoche)
    writer.add_scalar("Accuracy/Training", train_acc, epoche)
    writer.add_scalar("Accuracy/Validierung", val_acc, epoche)
    if epoche in {0, 25, 50, 75, 100}:
        print(
            f"{epoche:>6} {train_loss:>10.4f} {train_acc:>9.3f} "
            f"{val_loss:>9.4f} {val_acc:>8.3f}"
        )

writer.close()
print("\nTensorBoard-Log:")
print(log_dir)
print("\nStart:")
print("tensorboard --logdir", log_dir.parent)
