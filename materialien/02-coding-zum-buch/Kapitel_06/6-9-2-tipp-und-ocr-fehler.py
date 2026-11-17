# Tipp- und OCR-Fehler probabilistisch korrigieren.
from collections import Counter
import unicodedata

import spacy
from rapidfuzz import fuzz, process

# Deutsche Tokenisierung verwenden.
nlp = spacy.blank("de")

def normalisiere(text):
    # Unicode und Groß-/Kleinschreibung vereinheitlichen.
    return unicodedata.normalize("NFC", text).casefold()

# Kleines domänenspezifisches Referenzkorpus.
referenz = [
    "Die Pumpe zeigt starke Vibrationen.",
    "Die Pumpe läuft ruhig.",
    "Das Lager ist defekt.",
    "Das Lager muss geprüft werden.",
    "Der Sensor meldet einen Fehler.",
    "Der Sensor arbeitet zuverlässig.",
]

# Wörter und Wortpaare des Referenzkorpus zählen.
saetze = [
    [
        normalisiere(token.text)
        for token in nlp.make_doc(text)
        if token.is_alpha
    ]
    for text in referenz
]

haeufigkeit = Counter(
    wort
    for satz in saetze
    for wort in satz
)

bigramme = Counter(
    (links, rechts)
    for satz in saetze
    for links, rechts in zip(satz, satz[1:])
)

vokabular = sorted(haeufigkeit)
anzahl_woerter = sum(haeufigkeit.values())
anzahl_typen = len(vokabular)

def kandidat_score(kandidat, links, rechts, aehnlichkeit):
    # Wort- und Kontextwahrscheinlichkeiten mit Glättung berechnen.
    p_wort = (
        (haeufigkeit[kandidat] + 1)
        / (anzahl_woerter + anzahl_typen)
    )
    p_links = (
        (bigramme[(links, kandidat)] + 1)
        / (haeufigkeit[links] + anzahl_typen)
        if links else 1.0
    )
    p_rechts = (
        (bigramme[(kandidat, rechts)] + 1)
        / (haeufigkeit[kandidat] + anzahl_typen)
        if rechts else 1.0
    )
    return (
        (aehnlichkeit / 100) ** 2
        * p_wort
        * p_links
        * p_rechts
    )

def korrigiere(text):
    # Unbekannte Wörter anhand mehrerer Kandidaten korrigieren.
    doc = nlp.make_doc(text)
    ersetzungen = []
    for i, token in enumerate(doc):
        form = normalisiere(token.text)
        if not token.is_alpha or form in haeufigkeit:
            continue
        links = normalisiere(doc[i - 1].text) if i else ""
        rechts = (
            normalisiere(doc[i + 1].text)
            if i + 1 < len(doc)
            else ""
        )
        treffer = process.extract(
            form,
            vokabular,
            scorer=fuzz.ratio,
            limit=3,
        )
        scores = [
            (
                kandidat,
                kandidat_score(
                    kandidat,
                    links,
                    rechts,
                    aehnlichkeit,
                ),
            )
            for kandidat, aehnlichkeit, _ in treffer
        ]
        gesamt = sum(score for _, score in scores)
        kandidaten = [
            (kandidat, score / gesamt)
            for kandidat, score in scores
        ]
        kandidat, wahrscheinlichkeit = max(
            kandidaten,
            key=lambda eintrag: eintrag[1],
        )
        ersatz = (
            kandidat.capitalize()
            if token.text[0].isupper()
            else kandidat
        )
        ersetzungen.append(
            (
                token.idx,
                token.idx + len(token),
                token.text,
                ersatz,
                wahrscheinlichkeit,
            )
        )
    korrigiert = text
    # Von hinten ersetzen, damit Zeichenpositionen gültig bleiben.
    for start, ende, _, ersatz, _ in reversed(ersetzungen):
        korrigiert = (
            korrigiert[:start]
            + ersatz
            + korrigiert[ende:]
        )
    return korrigiert, ersetzungen

beispiele = {
    "Tippfehler": "Die Pumoe zeigt starke Vibrationen.",
    "OCR": "Das Lagcr ist defekt.",
}

for quelle, text in beispiele.items():
    korrigiert, ersetzungen = korrigiere(text)
    print(quelle)
    for _, _, alt, neu, p in ersetzungen:
        print(
            f"{alt} -> {neu} | Wahrscheinlichkeit: {p:.3f}"
        )

    print(f"Korrigiert: {korrigiert}\n")
