# Einen Wartungsbericht mit spaCy in Felder überführen
import spacy

# Leere deutsche Pipeline mit regelbasierter Entitätserkennung.
nlp = spacy.blank("de")
ruler = nlp.add_pipe("entity_ruler")
patterns = [
    {
        "label": "ANLAGE",
        "pattern": [{"LOWER": "pumpe"}, {"LIKE_NUM": True}],
    },
    {
        "label": "PROBLEM",
        "pattern": [
            {"LOWER": "starke"},
            {"LOWER": "vibrationen"},
        ],
    },
    {"label": "MASSNAHME", "pattern": [{"LOWER": "lager"}]},
    {
        "label": "FRIST",
        "pattern": [{"LOWER": "zwei"}, {"LOWER": "wochen"}],
    },
    {"label": "PRIORITAET", "pattern": [{"LOWER": "hoch"}]},
]
ruler.add_patterns(patterns)
text = (
    "Pumpe 4 weist starke Vibrationen auf. "
    "Lager innerhalb von zwei Wochen prüfen. Priorität hoch."
)
doc = nlp(text)
result = {ent.label_.lower(): ent.text for ent in doc.ents}
print("STRUKTURIERTE INFORMATIONEN")
for key in [
    "anlage",
    "problem",
    "massnahme",
    "frist",
    "prioritaet",
]:
    print(f"{key:<11}: {result.get(key, 'FEHLT')}")
