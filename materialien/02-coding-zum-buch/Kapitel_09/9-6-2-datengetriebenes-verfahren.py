# Zustandsbewertungen mit Wort- und Bigrammmerkmalen lernen
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

texte = [
    "Maschine läuft ruhig", "Anlage ist stabil",
    "Ventil ist dicht", "Motor ist funktionsfähig",
    "Lager ist nicht defekt", "Pumpe arbeitet zuverlässig",
    "Maschine läuft laut", "Anlage ist instabil",
    "Ventil ist undicht", "Motor ist defekt",
    "Ventil ist nicht dicht", "Pumpe läuft nicht ruhig",
]
labels = [
    "positiv", "positiv", "positiv", "positiv",
    "positiv", "positiv", "negativ", "negativ",
    "negativ", "negativ", "negativ", "negativ",
]
modell = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=500, random_state=7),
)
modell.fit(texte, labels)
test = [
    "Motor arbeitet stabil",
    "Das Ventil ist undicht",
    "Das Lager ist nicht defekt",
]
for text, label in zip(test, modell.predict(test)):
    print(f"{label:<8} {text}")
