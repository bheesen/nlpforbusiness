# Annotationen auf Grenzen, Labels und Überlappungen prüfen
text = "Pumpe 4 zeigt starke Vibrationen."
annotationen = [
    {"start": 0, "ende": 7, "label": "ANLAGE"},
    {"start": 15, "ende": 32, "label": "PROBLEM"},
    {"start": 29, "ende": 40, "label": "PROBLEM"},
]
erlaubt = {"ANLAGE", "PROBLEM", "MASSNAHME"}

for nr, ann in enumerate(annotationen, start=1):
    fehler = []
    if not 0 <= ann["start"] < ann["ende"] <= len(text):
        fehler.append("Grenze")
    if ann["label"] not in erlaubt:
        fehler.append("Label")
    vorher = annotationen[:nr - 1]
    if any(ann["start"] < a["ende"] and a["start"] < ann["ende"]
           for a in vorher):
        fehler.append("Überlappung")
    status = ", ".join(fehler) if fehler else "gültig"
    print(f"Annotation {nr}: {status}")
