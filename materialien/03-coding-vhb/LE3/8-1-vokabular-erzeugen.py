# Ein kontrolliertes Vokabular mit spaCy erzeugen
import spacy

nlp = spacy.blank("de")
texte = [
    "Pumpe 4 vibriert stark.",
    "Motor 2 wird sehr warm.",
    "Pumpe 4 verliert Öl.",
]

# Satzzeichen ausschließen und Kleinschreibung vereinheitlichen.
tokens = [
    [token.lower_ for token in nlp(text) if not token.is_punct]
    for text in texte
]
vokabular = sorted({token for doc in tokens for token in doc})
print("Texte:")
for nr, doc in enumerate(texte, start=1):
    print(f"Text-{nr}: {doc}")
print("\nTokenlisten:")
for nr, doc in enumerate(tokens, start=1):
    print(f"Token-{nr}: {doc}")
print("\nVokabular:")
print(vokabular)
print("\nMerkmale:", len(vokabular))
