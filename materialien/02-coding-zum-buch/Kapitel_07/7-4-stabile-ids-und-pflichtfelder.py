# Stabile IDs erzeugen und Pflichtfelder prüfen
import hashlib
import pandas as pd

berichte = pd.DataFrame([
    {"quelle": "Werk A", "datum": "2026-07-01",
     "text": "Pumpe 4 vibriert."},
    {"quelle": "Werk B", "datum": "2026-07-02",
     "text": "Motor 2 wird warm."},
])

basis = berichte["quelle"] + "|" + berichte["datum"] + "|" + berichte["text"]
berichte["id"] = basis.map(
    lambda wert: hashlib.sha256(wert.encode()).hexdigest()[:10]
)
pflicht = ["id", "quelle", "datum", "text"]
berichte["vollstaendig"] = berichte[pflicht].notna().all(axis=1)
print(berichte[pflicht + ["vollstaendig"]].to_string(index=False))
