# Empirische Wortwahrscheinlichkeiten aus Häufigkeiten schätzen
import spacy
from collections import Counter

# Leere deutsche Pipeline mit Tokenizer erzeugen.
nlp = spacy.blank("de")
texte = [
    "Die Pumpe vibriert stark.",
    "Die Pumpe verliert Öl.",
    "Der Motor wird heiß.",
    "Die Pumpe wird geprüft.",
]
# Nur alphabetische Tokens in die Wortstatistik übernehmen.
doc = nlp(" ".join(texte).lower())
tokens = [token.text for token in doc if token.is_alpha]
haeufigkeit = Counter(tokens)
gesamt = len(tokens)
print(f"Tokenzahl: {gesamt}")
print("Wort          Anzahl  Wahrscheinlichkeit")
print("-" * 42)
for wort, anzahl in haeufigkeit.most_common():
    p = anzahl / gesamt
    print(f"{wort:<13} {anzahl:>6}  {p:>18.4f}")
print(f"Summe: {sum(n / gesamt for n in haeufigkeit.values()):.4f}")
