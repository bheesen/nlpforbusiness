# Acht Plutchik-Basisemotionen mit dem NRC EmoLex bestimmen.
from collections import Counter
from pathlib import Path
BASIS = Path(__file__).resolve().parents[1]
DATEI = (BASIS / "EigeneDaten" / "NRC" / "NRC-Emotion-Lexicon-ForVariousLanguages.txt")
#%% Pakete importieren
import pandas as pd
import spacy
nlp = spacy.load("de_core_news_sm")
EMOTIONEN = ["anger", "anticipation", "disgust", "fear",
             "joy", "sadness", "surprise", "trust",]
DEUTSCH = {"anger": "Ärger", "anticipation": "Erwartung",
           "disgust": "Ekel", "fear": "Angst",
           "joy": "Freude", "sadness": "Traurigkeit",
           "surprise": "Überraschung", "trust": "Vertrauen",}
#%% Deutsches Emotionslexikon erzeugen
daten = pd.read_csv(DATEI, sep="\t"); lexikon = {}
for _, zeile in daten.iterrows():
    wort = str(zeile["German"]).strip().casefold()
    if not wort or wort == "nan":
        continue
    labels = {
        emotion
        for emotion in EMOTIONEN
        if int(zeile[emotion]) == 1
    }
    if labels:
        lexikon.setdefault(wort, set()).update(labels)
def analysieren(text):
    # Oberfläche und Lemma berücksichtigen.
    scores = Counter()
    fundstellen = {}
    for token in nlp(text):
        formen = {token.text.casefold(), token.lemma_.casefold()}
        labels = set()
        for form in formen:
            labels.update(lexikon.get(form, set()))
        for label in labels:
            name = DEUTSCH[label]
            scores[name] += 1
            fundstellen.setdefault(name, []).append(token.text)
    return scores, fundstellen
#%% Rückmeldungen analysieren
texte = ["Ich bin wütend und habe Angst vor einem erneuten Ausfall.",
    "Ich empfinde Freude und Vertrauen in den neuen Service.",]
for text in texte:
    scores, fundstellen = analysieren(text)
    print(f"\nText: {text}")
    print(f"{'Emotion':15s} {'Anzahl':>6s}  Erkannte Wörter")
    print("-" * 55)
    for name, score in scores.most_common():
        woerter = ", ".join(fundstellen[name])
        print(f"{name:15s} {score:6d}  {woerter}")

