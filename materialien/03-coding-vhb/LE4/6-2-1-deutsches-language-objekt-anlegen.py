# Grundstruktur einer spaCy-Verarbeitung untersuchen
import spacy
# Leere deutsche Sprachpipeline erzeugen.
nlp = spacy.blank("de")
text = "Pumpe 4 vibriert stark."
print("Text:" ,text)
# Text tokenisieren und als Doc bereitstellen.
doc = nlp(text)
print("Pipeline:", nlp.pipe_names)
print("Tokens:", [token.text for token in doc])


