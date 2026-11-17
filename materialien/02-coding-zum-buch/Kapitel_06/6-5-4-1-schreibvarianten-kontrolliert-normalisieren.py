# Schreibvarianten vereinheitlichen und Originaltext erhalten
import re
import unicodedata
def normalize(text: str) -> str:
    # Unicode, Leerraum, Datum und Einheit normalisieren
    text = unicodedata.normalize("NFC", text).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(\d+)\s*kg\b", r"\1 kg", text)
    return re.sub(r"(\d{2})\.(\d{2})\.(\d{4})", r"\3-\2-\1", text)
original = "  Prüfung am 03.08.2026: Gewicht 12KG.  "
print("Original:  ", original)
print("Normalisiert:", normalize(original))




