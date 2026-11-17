# Kontextpaare für unterschiedliche Fenstergrößen bilden
import spacy

# Leere deutsche spaCy-Pipeline zur Tokenisierung erzeugen.
nlp = spacy.blank("de")
satz = "techniker prüft lager und welle"
print("Satz:", satz)
tokens = [t.text for t in nlp.make_doc(satz) if t.is_alpha]
for fenster in [1, 2]:
    paare = []
    for i, wort in enumerate(tokens):
        links = max(0, i - fenster)
        rechts = min(len(tokens), i + fenster + 1)
        for j in range(links, rechts):
            if i != j:
                paare.append((wort, tokens[j]))
    print(f"\nFenstergröße {fenster}: {len(paare)} Paare")
    print(", ".join(f"{a}->{b}" for a, b in paare[:10]))
