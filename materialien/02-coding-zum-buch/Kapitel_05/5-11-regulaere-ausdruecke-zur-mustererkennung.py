# Strukturierte Angaben mit regulären Ausdrücken finden
import re

text = (
    "Pumpe P-17 wurde am 05.08.2026 geprüft. "
    "Kontakt: service@example.org"
)
anlage = re.search(r"\b[A-Z]-\d+\b", text)
datum = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", text)
mail = re.search(r"[\w.-]+@[\w.-]+", text)

print(f"Anlage: {anlage.group() if anlage else None}")
print(f"Datum: {datum.group() if datum else None}")
print(f"E-Mail: {mail.group() if mail else None}")