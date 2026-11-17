# Technische Qualitätsprobleme erkennen und klassifizieren.
import pandas as pd

daten = pd.DataFrame(
    {
        "id": ["B01", "B02", "B03", "B04", "B05"],
        "konfidenz": [0.94, 0.31, 0.88, 0.72, 0.46],
        "text": [
            "Pumpe 4 zeigt starke Vibrationen.",
            "???? ### !!!",
            "",
            (
                "Die Pumpe zeigt seit mehreren Stunden starke "
                "Vibrationen und soll vollständig geprüft werden."
            ),
            "Lagcr 7 ist dcfekt.",
        ],
    }
)

def sonderanteil(text):
    # Anteil ungewöhnlicher Sonderzeichen bestimmen.
    if not text:
        return 0.0
    sonder = sum(
        not z.isalnum() and not z.isspace()
        for z in text
    )
    return sonder / len(text)

def fehlerklassen(zeile):
    # Mehrere Qualitätsprobleme je Dokument zulassen.
    fehler = []
    text = zeile["text"].strip()
    if not text:
        fehler.append("LEER")
    if sonderanteil(text) >= 0.20:
        fehler.append("SONDERZEICHEN")
    if zeile["konfidenz"] < 0.60:
        fehler.append("NIEDRIGE_KONFIDENZ")
    if len(text) >= 80:
        fehler.append("SEHR_LANG")
    return fehler

daten["fehler"] = daten.apply(fehlerklassen, axis=1)
auffaellig = daten[daten["fehler"].str.len() > 0].copy()

print("AUFFÄLLIGE DOKUMENTE")
for _, zeile in auffaellig.iterrows():
    klassen = ", ".join(zeile["fehler"])
    print(f"{zeile['id']}: {klassen}")
print("\nFEHLERHÄUFIGKEIT")
haeufigkeit = (
    auffaellig["fehler"]
    .explode()
    .value_counts()
)
print(haeufigkeit.to_string())