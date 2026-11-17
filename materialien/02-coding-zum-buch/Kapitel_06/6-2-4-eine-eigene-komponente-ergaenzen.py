# Eine eigene Pipeline-Komponente ergänzen
import unicodedata
import spacy
from spacy.language import Language

# Schreibweise für Vergleiche vereinheitlichen.
def normalisiere(text):
    return unicodedata.normalize("NFC", text).casefold()

begriffe = {
    normalisiere(text)
    for text in {"Pumpe", "Lager", "Ventil", "Ölpumpe"}
}

@Language.component("wartungsmerkmal")
def wartungsmerkmal(doc):
    # Wartungsbegriffe unabhängig von Schreibweise erkennen.
    doc.user_data["wartung"] = any(
        normalisiere(token.text) in begriffe
        for token in doc
    )
    return doc

nlp = spacy.blank("de")
nlp.add_pipe("wartungsmerkmal")

text = "Die ÖLPUMPE und das lager werden geprüft."
doc = nlp(text)

print("Pipeline:", nlp.pipe_names)
print("Wartungsbezug:", doc.user_data["wartung"])