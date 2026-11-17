# Wiederverwendbare Schritte einer kleinen Textpipeline
import re

def normalisiere(text: str) -> str:
    """Vereinheitlicht Leerraum und Groß-/Kleinschreibung."""
    text = re.sub(r"\s+", " ", text.strip())
    return text.casefold()

def tokenisiere(
    text: str,
    tokens: list[str] | None = None
) -> list[str]:
    """Ergänzt eine neue oder übergebene Tokenliste."""
    if tokens is None:
        tokens = []

    neue_tokens = re.findall(
        r"\w+(?:-\w+)*",
        normalisiere(text)
    )
    tokens.extend(neue_tokens)
    return tokens

text = "  Sensor S-2   liefert FEHLER.  "

print(f"Normalisiert: {normalisiere(text)}")
print(f"Tokens: {tokenisiere(text)}")
