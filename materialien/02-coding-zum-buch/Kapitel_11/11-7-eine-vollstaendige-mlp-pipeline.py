# Wartungsberichte mit einem bekannten MLP klassifizieren.
#%% Verzeichnisse definieren
from pathlib import Path
import sys
BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS
#%% Pakete importieren
from nlp_eda import KLASSEN, UNK, bow_daten
from nlp_eda import setze_seed, tokenisiere, trainiere
import torch
#%% Trainingsdaten laden und Modell definieren
setze_seed()
woerter, index, x, y = bow_daten()
modell = torch.nn.Sequential(
    torch.nn.Linear(len(woerter), 12),
    torch.nn.ReLU(),
    torch.nn.Linear(12, len(KLASSEN)),)
trainiere(modell, x, y)
#%% Neue Texte wie die Trainingsdaten vektorisieren
def vektorisiere(text):
    # Wörter zählen und in den bekannten Merkmalsraum einordnen.
    vektor = torch.zeros(len(woerter))
    for wort in tokenisiere(text):
        pos = index.get(wort, index[UNK])
        vektor[pos] += 1
    return vektor
tests = ["pumpe ist undicht",
         "encoder meldet falsche position",
         "motorlager vibriert",
         "anlage zeigt unbekanntes problem",]
#%% Neue Wartungsmeldungen klassifizieren
modell.eval()
with torch.no_grad():
    for text in tests:
        logits = modell(vektorisiere(text))
        probs = torch.softmax(logits, dim=0)
        wert, klasse = probs.max(0)
        status = ("automatisch"
                   if wert >= 0.70
                    else "manuell prüfen")
        print(f"{wert.item():.3f} "
              f"{KLASSEN[klasse]} | {status}")
        print(" ", text)
        
        