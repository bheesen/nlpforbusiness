# Ein kleines Korpus mit Text und Metadaten beschreiben
import pandas as pd

berichte = [
    {"id": "B01", "anlage": "Pumpe", "text": "Lager ist laut."},
    {"id": "B02", "anlage": "Motor", "text": "Motor wird warm."},
    {"id": "B03", "anlage": "Pumpe", "text": "Dichtung ist undicht."},
]

df = pd.DataFrame(berichte)
df["zeichen"] = df["text"].str.len()
print(df.to_string(index=False))
print("\nDokumente:", len(df))
print("Anlagen:", df["anlage"].value_counts().to_dict())
