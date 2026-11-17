# Dokumente zeilenweise als JSONL speichern und lesen
import json
from pathlib import Path

dokumente = [
    {"id": "B01", "text": "Pumpe 4 vibriert."},
    {"id": "B02", "text": "Sensor S-2 prüfen."},
]
pfad = Path("wartung.jsonl")

with pfad.open("w", encoding="utf-8") as datei:
    for dokument in dokumente:
        zeile = json.dumps(dokument, ensure_ascii=False)
        datei.write(zeile + "\n")

with pfad.open(encoding="utf-8") as datei:
    geladen = [json.loads(zeile) for zeile in datei]

print(f"Datei: {pfad.name}")
print(f"Dokumente: {len(geladen)}")
print(f"Zweiter Text: {geladen[1]['text']}")
