# TXT, CSV und JSONL in ein gemeinsames Schema überführen
import json
from io import StringIO

import pandas as pd

text = "Pumpe 4 vibriert stark."
csv_text = "id,text\nB02,Motor 2 wird warm.\n"
jsonl = '{"id":"B03","text":"Ventil 1 klemmt."}'

zeilen = [{"id": "B01", "text": text, "format": "TXT"}]
zeilen += pd.read_csv(StringIO(csv_text)).assign(format="CSV").to_dict(
    "records"
)
objekt = json.loads(jsonl)
objekt["format"] = "JSONL"
zeilen.append(objekt)
print(pd.DataFrame(zeilen).to_string(index=False))
