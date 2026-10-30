# Doc, Token und Span untersuchen
import spacy
nlp = spacy.blank("de")
doc = nlp("Techniker Müller prüft Pumpe 4.")
# Eigenschaften aller Tokens ausgeben.
print("Token-Informationen")
print(f"{'Nr.':>3} "
      f"{'Token':<12} "
      f"{'Offset':>6} "
      f"{'Alpha':>7} "
      f"{'Zahl':>6}")
print("-" * 38)
for token in doc:
    print(f"{token.i:>3} "
          f"{token.text:<12} "
          f"{token.idx:>6} "
          f"{str(token.is_alpha):>7} "
          f"{str(token.like_num):>6}")
# Zwei Tokens als Span zusammenfassen.
span = doc[0:2]
print("\nSpan")
print(f"Text: {span.text}")


