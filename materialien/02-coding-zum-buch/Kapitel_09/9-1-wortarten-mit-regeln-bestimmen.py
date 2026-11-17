# Wortarten mit einer trainierten spaCy-Pipeline bestimmen
import spacy

# Deutsches Sprachmodell laden
nlp = spacy.load("de_core_news_sm")
# Deutsche Bezeichnungen der Universal-POS-Tags
pos_de = {
    "ADJ": "Adjektiv",
    "ADP": "Adposition",
    "ADV": "Adverb",
    "AUX": "Hilfsverb",
    "CCONJ": "nebenordnende Konjunktion",
    "DET": "Determinierer",
    "INTJ": "Interjektion",
    "NOUN": "Substantiv",
    "NUM": "Zahlwort",
    "PART": "Partikel",
    "PRON": "Pronomen",
    "PROPN": "Eigenname",
    "PUNCT": "Satzzeichen",
    "SCONJ": "unterordnende Konjunktion",
    "SYM": "Symbol",
    "VERB": "Verb",
    "X": "sonstige Einheit",
}
text = "Der Techniker prüft die Pumpe heute sorgfältig."
print("Text: ", text)
# Text mit spaCy analysieren
doc = nlp(text)
print(f"\n{'Token':<14} {'POS':<7} Deutsche Bezeichnung")
print("-" * 52)
for token in doc:
    # Universal-POS-Tag des Tokens auslesen
    pos = token.pos_
    # Deutsche Bezeichnung zum POS-Tag nachschlagen
    deutsch = pos_de.get(pos, "unbekannt")
    print(f"{token.text:<14} {pos:<7} {deutsch}")
    
    
