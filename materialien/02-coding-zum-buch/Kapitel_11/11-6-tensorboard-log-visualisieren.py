# TensorBoard-Protokolle lesen und als PNG-Dateien darstellen.
#%% Verzeichnisse definieren
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

BASIS = Path(
    r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP"
)
EigenerOutput = BASIS / "EigenerOutput"

#%% TensorBoard-Protokoll laden
log_verzeichnis = EigenerOutput / "tensorboard"

if not log_verzeichnis.exists():
    raise FileNotFoundError(
        f"TensorBoard-Verzeichnis fehlt: {log_verzeichnis}"
    )

# Alle protokollierten Skalarwerte einlesen.
protokoll = EventAccumulator(
    str(log_verzeichnis),
    size_guidance={"scalars": 0},
)
protokoll.Reload()

# Erwartete TensorBoard-Tags prüfen.
tags = set(protokoll.Tags()["scalars"])
erwartet = {
    "Loss/Training",
    "Loss/Validierung",
    "Accuracy/Training",
    "Accuracy/Validierung",
}
fehlend = erwartet - tags

if fehlend:
    raise ValueError(
        "Folgende TensorBoard-Tags fehlen: "
        + ", ".join(sorted(fehlend))
    )

#%% Hilfsfunktion zum Auslesen einer Skalarreihe
def werte(tag):
    """Epochen und Werte eines TensorBoard-Tags zurückgeben."""
    eintraege = protokoll.Scalars(tag)
    epochen = [eintrag.step for eintrag in eintraege]
    messwerte = [eintrag.value for eintrag in eintraege]
    return epochen, messwerte


#%% Loss visualisieren
ep_train, loss_train = werte("Loss/Training")
ep_val, loss_val = werte("Loss/Validierung")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ep_train, loss_train, label="Training")
ax.plot(ep_val, loss_val, label="Validierung")
ax.set_xlabel("Epoche")
ax.set_ylabel("Loss")
ax.set_title("Trainings- und Validierungs-Loss")
ax.legend()
fig.tight_layout()

loss_datei = EigenerOutput / "tensorboard_loss.png"
fig.savefig(loss_datei, dpi=200)
plt.show()
plt.close(fig)

#%% Accuracy visualisieren
ep_train, acc_train = werte("Accuracy/Training")
ep_val, acc_val = werte("Accuracy/Validierung")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ep_train, acc_train, label="Training")
ax.plot(ep_val, acc_val, label="Validierung")
ax.set_xlabel("Epoche")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Trainings- und Validierungs-Accuracy")
ax.legend()
fig.tight_layout()

acc_datei = EigenerOutput / "tensorboard_accuracy.png"
fig.savefig(acc_datei, dpi=200)
plt.show()
plt.close(fig)

#%% Ergebnis ausgeben
print("Grafikdateien erzeugt:")
print(loss_datei)
print(acc_datei)
