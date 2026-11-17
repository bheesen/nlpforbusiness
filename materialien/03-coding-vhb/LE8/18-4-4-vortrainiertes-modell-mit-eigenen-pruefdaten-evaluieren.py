# Ein vortrainiertes Sentimentmodell mit eigenen Prüfdaten evaluieren.
#%% Verzeichnisse definieren
from pathlib import Path
import sys
BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS
#%% Pakete importieren
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import pipeline
MODELL = "oliverguhr/german-sentiment-bert"
# Prüfdaten einlesen und Sentimentmodell initialisieren.
daten = pd.read_csv(EigeneDaten / "feedback_test.csv")
geraet = 0 if torch.cuda.is_available() else -1
sentiment = pipeline(
    "text-classification",
    model=MODELL,
    tokenizer=MODELL,
    device=geraet,
)
# Prüfdaten klassifizieren, ohne das Modell zu trainieren.
ergebnisse = sentiment(daten["text"].tolist())
pred = [ergebnis["label"] for ergebnis in ergebnisse]
# Vorhersage, tatsächliches Label und Text gegenüberstellen.
vergleich = daten[["label", "text"]].copy()
vergleich.insert(0, "Vorhersage", pred)
vergleich.columns = ["Vorhersage", "Label", "Text"]
print("Vergleich der Sentimentklassifikation:")
print(vergleich.to_string(index=False))
# Klassifikationsbericht ausgeben.
print("\nKlassifikationsbericht:")
print(classification_report(daten["label"], pred, digits=3))
