# NLP-Daten mit Python-Collections organisieren
from collections import Counter
# Ausgangstext
text = ("Pumpe 4 zeigt Vibrationen. "
        "Wir sollten die Pumpe stoppen.")
# Text vereinheitlichen und in Tokens zerlegen
tokens = text.casefold().replace(".", "").split()
# Eindeutige Wörter und Worthäufigkeiten bestimmen
vokabular = set(tokens)
haeufigkeit = Counter(tokens)
# Den Ausdruck "Pumpe 4" als Anlage markieren
span = (0, 7, "ANLAGE")
# Text und Annotation gemeinsam speichern
dokument = {
    "id": "B01",
    "text": text,
    "label": "kritisch",
    "spans": [span]
}
# Markierten Textabschnitt auslesen
start, ende, typ = span
fundstelle = dokument["text"][start:ende]
print(f"Text: {text}")
print(f"\nTokens: {tokens}")
print(f"Vokabular: {sorted(vokabular)}")
print(f"Häufigkeiten: {haeufigkeit.most_common()}")
print(f"\nFundstelle: {fundstelle}")
print(f"Entitätstyp: {typ}")
print(f"Dokumentlabel: {dokument['label']}")

