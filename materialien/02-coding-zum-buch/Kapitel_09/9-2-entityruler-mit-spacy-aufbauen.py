# Einen regelbasierten EntityRuler mit spaCy einsetzen
import spacy
nlp = spacy.blank("de")
ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns([
    {
        "label": "ANLAGE",
        "pattern": [
            {"LOWER": {"IN": ["pumpe", "motor", "ventil"]}},
            {"IS_DIGIT": True},
        ],
    },
    {
        "label": "PRIORITAET",
        "pattern": [
            {"LOWER": "priorität"},
            {"LOWER": {"IN": ["hoch", "mittel", "niedrig"]}},
        ],
    },
    {
        "label": "FRIST",
        "pattern": [
            {"LOWER": "innerhalb"},
            {"LOWER": "von"},
            {"LOWER": {"REGEX": "^(?:\\d+|ein|eine|zwei|drei)$"}},
            {"LOWER": {"IN": ["tagen", "wochen"]}},
        ],
    },
])
text = ("Pumpe 4 prüfen, Priorität hoch, innerhalb von zwei Wochen.")
print("Text: ", text, "\n")
doc = nlp(text)
print(f"{'Entität':<25} {'Label':<12} {'Position'}")
print("-" * 48)
for ent in doc.ents:
    print(f"{ent.text:<25} {ent.label_:<12} {ent.start}:{ent.end}")
    
    
