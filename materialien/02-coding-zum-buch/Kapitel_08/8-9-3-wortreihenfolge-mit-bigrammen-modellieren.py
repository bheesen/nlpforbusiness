#%% Ein Bigramm-Modell macht die Wortreihenfolge sichtbar
from collections import Counter
import math
import spacy

# Leere deutsche spaCy-Pipeline für die Tokenisierung erzeugen.
nlp = spacy.blank("de")

# Denselben Korpus wie in den vorangehenden Übungen verwenden.
saetze = [
    "die pumpe vibriert stark",
    "die pumpe verliert öl",
    "der motor wird heiß",
    "die pumpe wird geprüft",
]

def tokenisiere(text):
    # Nur alphabetische Tokens für das Sprachmodell verwenden.
    return [
        token.text
        for token in nlp(text)
        if token.is_alpha
    ]

# Häufigkeiten der Vorgänger und der Bigramme speichern.
uni = Counter()
bi = Counter()
for satz in saetze:
    # Satzanfang und Satzende als eigene Symbole ergänzen.
    woerter = ["<s>"] + tokenisiere(satz) + ["</s>"]

    # Jeder Vorgänger wird für den Nenner gezählt.
    uni.update(woerter[:-1])

    # Je zwei aufeinanderfolgende Tokens bilden ein Bigramm.
    bi.update(zip(woerter[:-1], woerter[1:]))

def bigramm_p(vorher, wort):
    # Bedingte Wahrscheinlichkeit P(wort | vorher) berechnen.
    if uni[vorher] == 0:
        return 0.0
    return bi[vorher, wort] / uni[vorher]

def satz_p(satz):
    # Satz tokenisieren und Start- sowie Endsymbol ergänzen.
    woerter = ["<s>"] + tokenisiere(satz) + ["</s>"]
    # Mit Wahrscheinlichkeit 1 starten.
    wahrscheinlichkeit = 1.0
    # Wahrscheinlichkeiten aller Bigramme multiplizieren.
    for vorher, wort in zip(woerter[:-1], woerter[1:]):
        wahrscheinlichkeit *= bigramm_p(vorher, wort)
    return wahrscheinlichkeit

def satz_logp(satz):
    # Bigramm-Wahrscheinlichkeiten des Satzes bestimmen.
    woerter = ["<s>"] + tokenisiere(satz) + ["</s>"]
    werte = [
        bigramm_p(vorher, wort)
        for vorher, wort in zip(woerter[:-1], woerter[1:])
    ]
    # Ein unbeobachtetes Bigramm führt ohne Glättung zu -unendlich.
    if 0.0 in werte:
        return float("-inf")
    # Das Produkt wird im Log-Raum zur Summe.
    return sum(math.log2(wert) for wert in werte)

# Beobachteten Satz und dieselben Wörter in anderer Reihenfolge prüfen.
tests = [
    "die pumpe vibriert stark",
    "pumpe die stark vibriert",
]
for satz in tests:
    print(f"\nSATZ: {satz}")
    print("-" * 54)
    print(f"{'Bigramm':<30} {'Wahrscheinlichkeit':>20}")
    print("-" * 54)
    # Einzelne Bigramm-Wahrscheinlichkeiten sichtbar machen.
    woerter = ["<s>"] + tokenisiere(satz) + ["</s>"]
    for vorher, wort in zip(woerter[:-1], woerter[1:]):
        p = bigramm_p(vorher, wort)
        name = f"{vorher} -> {wort}"
        print(f"{name:<30} {p:>20.4f}")
    print("-" * 54)
    direkt = satz_p(satz)
    logp = satz_logp(satz)
    # Minus unendlich lesbar darstellen.
    if logp == float("-inf"):
        log_ausgabe = "-unendlich"
    else:
        log_ausgabe = f"{logp:.4f}"
    print(f"Satzwahrscheinlichkeit:{direkt:>28.4f} "
        f"\nSatzwahrscheinlichkeit(log):           {log_ausgabe:>12}")
