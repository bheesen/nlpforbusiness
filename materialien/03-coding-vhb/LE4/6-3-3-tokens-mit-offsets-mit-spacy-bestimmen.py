# Tokens und ihre Positionen im Originaltext bestimmen
import spacy
# Leere deutsche Pipeline mit Tokenizer erzeugen.
nlp = spacy.blank("de")
text = "E-Mail an service@example.org senden!"
print("Text: ", text, "\n")
doc = nlp(text)
# Token sowie Start- und Endposition ausgeben.
for token in doc:
    start = token.idx
    ende = start + len(token)
    print(f"{start:>2}:{ende:<2}  {token.text}")

