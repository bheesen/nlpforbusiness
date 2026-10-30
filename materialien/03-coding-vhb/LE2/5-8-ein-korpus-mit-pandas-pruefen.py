# Ein kleines Textkorpus mit pandas untersuchen
import pandas as pd

daten = [
    {"id": "B01", "text": "Pumpe 4 vibriert.", "label": "kritisch"},
    {"id": "B02", "text": "Sensor prüfen.", "label": "normal"},
    {"id": "B03", "text": None, "label": "normal"},
    {"id": "B04", "text": "Sensor prüfen.", "label": "normal"},
]
df = pd.DataFrame(daten)
df["zeichen"] = df["text"].fillna("").str.len()
df["text_fehlt"] = df["text"].isna()
bereinigt = df.dropna(subset=["text"]).drop_duplicates("text")
print(bereinigt[["id", "label", "zeichen"]].to_string(index=False))
print("\nDokumente je Label:")
print(bereinigt.groupby("label").size().to_string())

