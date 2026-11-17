# Vortrainiertes deutsches BERT-Modell für Sentiment verwenden.
#%% Pakete importieren
import torch
from transformers import pipeline
MODELL = "oliverguhr/german-sentiment-bert"
HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Falls vorhanden, die GPU für die Inferenz verwenden.
geraet = 0 if torch.cuda.is_available() else -1
sentiment = pipeline(
    "text-classification",
    model=MODELL,
    tokenizer=MODELL,
    device=geraet,
)
# Das Modell wird direkt genutzt und nicht mit eigenen Beispielen trainiert.
texte = [
    "Die Störung wurde schnell und dauerhaft behoben.",
    "Wir warten seit Tagen auf eine brauchbare Lösung.",
    "Die Lösung ist nicht schlecht.",
    "Der Termin wurde auf Donnerstag verschoben.",
]
# Für jeden Text die Wahrscheinlichkeiten aller Klassen ausgeben.
for text in texte:
    ergebnisse = sentiment(
        text,
        top_k=None,
    )
    print("\nText:", text)
    for ergebnis in ergebnisse:
        label = ergebnis["label"]
        score = ergebnis["score"]
        print(f"{label:8s}: {score:.4f}")
