# Personen- und Kontaktdaten durch Platzhalter ersetzen
import re

text = (
    "Technikerin Anna Weber meldet sich unter anna@werk.de. "
    "Rückruf: 030 1234567."
)
muster = [
    (r"\b(?:Technikerin|Techniker) [A-ZÄÖÜ][a-zäöüß]+ "
     r"[A-ZÄÖÜ][a-zäöüß]+\b", "<PERSON>"),
    (r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "<E-MAIL>"),
    (r"\b0\d{2,4}[ -]?\d{6,8}\b", "<TELEFON>"),
]
anonym = text
for regex, ersatz in muster:
    anonym = re.sub(regex, ersatz, anonym)
print("Original:")
print(text)
print("\nMaskiert:")
print(anonym)
