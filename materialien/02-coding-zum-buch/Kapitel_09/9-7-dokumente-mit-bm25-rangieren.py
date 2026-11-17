# Dokumente mit spaCy-Tokenisierung und BM25 durchsuchen
from collections import Counter
from math import log

import spacy

nlp = spacy.blank("de")
texte = [
    "Pumpe zeigt starke Vibrationen am Lager",
    "Motor erreicht eine hohe Temperatur",
    "Lager der Pumpe läuft laut und unruhig",
    "Ventil verliert Öl an der Dichtung",
]
anfrage = "laute Vibration Pumpe"

def tokenisiere(text):
    # Satzzeichen entfernen und Kleinschreibung vereinheitlichen.
    return [t.lower_ for t in nlp(text) if not t.is_punct]

docs = [tokenisiere(text) for text in texte]
query = tokenisiere(anfrage)
mittel = sum(map(len, docs)) / len(docs)
k1, b = 1.5, 0.75
# Dokumenthäufigkeit jedes Suchterms nur einmal berechnen.
df = {term: sum(term in doc for doc in docs) for term in set(query)}

def score(doc):
    # BM25 kombiniert Termhäufigkeit, Seltenheit und Dokumentlänge.
    zaehler = Counter(doc)
    wert = 0.0
    for term in query:
        idf = log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
        tf = zaehler[term]
        nenner = tf + k1 * (1 - b + b * len(doc) / mittel)
        wert += idf * tf * (k1 + 1) / nenner if tf else 0
    return wert

werte = [score(doc) for doc in docs]
for rang, nr in enumerate(
    sorted(range(len(texte)), key=werte.__getitem__, reverse=True), start=1
):
    print(f"{rang}. {werte[nr]:.3f}  {texte[nr]}")
