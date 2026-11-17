# Beziehungen und ein Wartungsereignis extrahieren

import json
import spacy
from spacy.matcher import DependencyMatcher

# Deutsches Modell für die linguistische Analyse laden.
nlp = spacy.load("de_core_news_sm")

# Fachspezifische Entitäten regelbasiert ergänzen.
ruler = nlp.add_pipe(
    "entity_ruler",
    after="ner",
    config={"overwrite_ents": True},
)

muster = [
    {
        "label": "PERSON",
        "pattern": [
            {"LOWER": "techniker"},
            {"POS": "PROPN"},
        ],
    },
    {
        "label": "TEIL",
        "pattern": [{"LEMMA": "Lager"}],
    },
    {
        "label": "ANLAGE",
        "pattern": [
            {"LOWER": "pumpe"},
            {"LIKE_NUM": True},
        ],
    },
    {
        "label": "PROBLEM",
        "pattern": [
            {"POS": "ADJ"},
            {"LEMMA": "Vibration"},
        ],
    },
    {
        "label": "ZEIT",
        "pattern": [
            {"LIKE_NUM": True},
            {"LOWER": "mai"},
        ],
    },
    {
        "label": "ORT",
        "pattern": [
            {"LOWER": "halle"},
            {"LIKE_NUM": True},
        ],
    },
]

ruler.add_patterns(muster)

# Subjekt und Objekt des Ereignisauslösers suchen.
matcher = DependencyMatcher(nlp.vocab)

ereignismuster = [
    {
        "RIGHT_ID": "trigger",
        "RIGHT_ATTRS": {"LEMMA": "prüfen"},
    },
    {
        "LEFT_ID": "trigger",
        "REL_OP": ">",
        "RIGHT_ID": "akteur",
        "RIGHT_ATTRS": {"DEP": "sb"},
    },
    {
        "LEFT_ID": "trigger",
        "REL_OP": ">",
        "RIGHT_ID": "objekt",
        "RIGHT_ATTRS": {"DEP": "oa"},
    },
]

matcher.add("PRUEFUNG", [ereignismuster])

# Wartungsbericht analysieren.
text = (
    "Am 12. Mai prüfte Techniker Müller das Lager der Pumpe 4 "
    "in Halle 2 wegen starker Vibrationen."
)

doc = nlp(text)

# Erkannte Entitäten nach ihrer Rolle speichern.
werte = {ent.label_: ent.text for ent in doc.ents}


def entitaet(token, label):
    # Die Entität ermitteln, die das Token enthält.
    return next(
        ent.text
        for ent in doc.ents
        if ent.label_ == label
        and ent.start <= token.i < ent.end
    )

# Ereignisauslöser, Akteur und Objekt bestimmen.
_, ids = matcher(doc)[0]
trigger, akteur, objekt = (doc[i] for i in ids)

person = entitaet(akteur, "PERSON")
teil = entitaet(objekt, "TEIL")

# Beziehungen als Tripel darstellen.
relationen = [
    (person, "PRUEFT", teil),
    (teil, "TEIL_VON", werte["ANLAGE"]),
    (
        werte["ANLAGE"],
        "HAT_PROBLEM",
        werte["PROBLEM"],
    ),
]

print("RELATIONEN")

for quelle, relation, ziel in relationen:
    print(f"({quelle}) -[{relation}]-> ({ziel})")

# Informationen zusätzlich zu einem Ereignis bündeln.
ereignis = {
    "typ": "PRUEFUNG",
    "ausloeser": trigger.text,
    "akteur": person,
    "objekt": teil,
    "anlage": werte["ANLAGE"],
    "problem": werte["PROBLEM"],
    "zeitpunkt": werte["ZEIT"],
    "ort": werte["ORT"],
    "status": None,
}

print("\nEREIGNIS")
print(json.dumps(ereignis, ensure_ascii=False, indent=2))
