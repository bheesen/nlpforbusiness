# Zustandsaussagen mit spaCy und Negationsregel bewerten
import spacy
nlp = spacy.blank("de")
positiv = {"stabil", "ruhig", "dicht", "funktionsfähig"}
negativ = {"defekt", "laut", "undicht", "heiß", "kritisch"}
def bewertung(text):
    # spaCy tokenisiert; Satzzeichen werden nicht bewertet.
    tokens = [t for t in nlp(text) if not t.is_punct]
    wert = 0
    for i, token in enumerate(tokens):
        faktor = -1 if i and tokens[i - 1].lower_ == "nicht" else 1
        if token.lower_ in positiv:
            wert += faktor
        if token.lower_ in negativ:
            wert -= faktor
    label = "positiv" if wert > 0 else "negativ" if wert < 0 else "neutral"
    return wert, label
texte = [
    "Das Lager läuft ruhig.",
    "Das Ventil ist undicht.",
    "Der Motor ist nicht defekt.",
    "Die Pumpe ist keineswegs stabil.",
]
for text in texte:
    wert, label = bewertung(text)
    print(f"{wert:>2}  {label:<8} {text}")
