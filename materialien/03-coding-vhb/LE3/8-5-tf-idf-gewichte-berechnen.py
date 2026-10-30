# TF-IDF-Gewichte für häufige und seltene Wörter berechnen
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

texte = [
    "Pumpe zeigt starke Vibration",
    "Motor zeigt hohe Temperatur",
    "Pumpe zeigt Ölverlust",
]

print("Texte:")
for nr, doc in enumerate(texte, start=1):
    print(f"Text-{nr}: {doc}")
vektorisierer = TfidfVectorizer()
matrix = vektorisierer.fit_transform(texte)
tabelle = pd.DataFrame(
    matrix.toarray(),
    columns=vektorisierer.get_feature_names_out(),
    index=["Text-1", "Text-2", "Text-3"],
)
print("\n",tabelle.round(3).to_string())
