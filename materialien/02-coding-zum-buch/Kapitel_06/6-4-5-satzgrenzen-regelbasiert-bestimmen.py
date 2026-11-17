# Satzgrenzen regelbasiert bestimmen und evaluieren.
import spacy
from spacy.language import Language

@Language.component("satzgrenzen")
def setze_satzgrenzen(doc):
    # Zunächst alle Tokens als Satzfortsetzung markieren.
    for token in doc:
        token.is_sent_start = False
    if len(doc):
        doc[0].is_sent_start = True

    # Satzgrenzen anhand der erzeugten Tokens setzen.
    for i, token in enumerate(doc[:-1]):
        if token.text in {".", "!", "?"}:
            doc[i + 1].is_sent_start = True
        elif token.text.endswith("."):
            if token.text[:-1].isdigit():
                doc[i + 1].is_sent_start = True
    return doc

# Leere deutsche Pipeline mit eigener Komponente erzeugen.
nlp = spacy.blank("de")
nlp.add_pipe("satzgrenzen")

text = (
    "Dr. Meier prüft Pumpe 4. "
    "Sie läuft z. B. nachts ruhig. Ende?"
)

doc = nlp(text)

print("TOKENS")
print("-" * 30)
for nummer, token in enumerate(doc):
    print(f"{nummer:>2}: {token.text}")

print("\nERKANNTE SÄTZE")
print("-" * 30)
for nummer, satz in enumerate(doc.sents, 1):
    print(f"{nummer}: {satz.text}")

# Referenzgrenzen als Zeichenpositionen der Folgesätze.
referenz = {25, 55}
erkannt = {
    satz.start_char
    for satz in doc.sents
    if satz.start_char > 0
}

tp = len(erkannt & referenz)
fp = len(erkannt - referenz)
fn = len(referenz - erkannt)

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0

print("\nEVALUATION")
print(f"Referenzgrenzen: {sorted(referenz)}")
print(f"Erkannte Grenzen: {sorted(erkannt)}")
print(f"TP={tp}, FP={fp}, FN={fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")