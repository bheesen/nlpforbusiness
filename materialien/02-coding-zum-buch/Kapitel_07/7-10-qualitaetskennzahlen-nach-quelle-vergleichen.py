# Qualitätskennzahlen nach Quelle vergleichen
import hashlib
import pandas as pd

korpus = pd.DataFrame([
    {"quelle": "Werk A", "text": "Pumpe ist laut."},
    {"quelle": "Werk A", "text": "Pumpe ist laut."},
    {"quelle": "Werk A", "text": ""},
    {"quelle": "Werk B", "text": "Motor wird sehr warm."},
    {"quelle": "Werk B", "text": "Ventil klemmt gelegentlich."},
])
korpus["leer"] = korpus["text"].str.strip().eq("")
korpus["zeichen"] = korpus["text"].str.len()
korpus["hash"] = korpus["text"].map(
    lambda text: hashlib.sha256(text.encode()).hexdigest()[:6]
)
korpus["dublette"] = korpus.duplicated("hash", keep=False)
bericht = korpus.groupby("quelle").agg(
    dokumente=("text", "size"), leer=("leer", "sum"),
    dubletten=("dublette", "sum"), mittlere_laenge=("zeichen", "mean"),
)
print(bericht.round(1).to_string())
