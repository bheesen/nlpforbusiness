# Zeichenbasierte Entitäten in BIO-Tags überführen
import spacy
# Deutsche Tokenisierung ohne trainiertes Modell vorbereiten.
nlp = spacy.blank("de")
text = "Pumpe 4 zeigt starke Vibrationen."
print("Text: ", text)
doc = nlp(text)
# Zeichenbereich nur bei exakter Übereinstimmung übernehmen.
ent = doc.char_span(0, 7, label="ANLAGE", alignment_mode="strict")
if ent is None:
    raise ValueError("Span liegt nicht auf Tokengrenzen.")
print(f"{'Token':12} {'Position':8} {'NER-Tag'}")
print("-" * 32)
doc.ents = [ent]
for token in doc:
    start = token.idx
    ende = start + len(token)
    tag = token.ent_iob_
    if token.ent_type_:
        tag = f"{tag}-{token.ent_type_}"
    print(f"{token.text:12}    {start:2}-{ende:2} {tag}")
    
    
