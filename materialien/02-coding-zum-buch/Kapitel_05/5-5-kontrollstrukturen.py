# Texte mit Bedingungen und Schleifen prüfen
berichte = [
    "Pumpe 4 sofort stoppen.",
    "  ",
    "Sensor S-2 bei Gelegenheit prüfen.",
]

for nummer, bericht in enumerate(berichte, start=1):
    text = bericht.strip()
    if not text:
        status = "übersprungen"
    elif "sofort" in text.casefold():
        status = "kritisch"
    else:
        status = "prüfen"
    print(f"{nummer}: {status:12} | {text or '<leer>'}")

#%%

# Parallele Sequenzen mit Wartungstexten und Labels
texte = [
    "Pumpe 4 zeigt starke Vibrationen.",
    "Ventil 2 arbeitet wieder normal.",
    "Lager 7 muss geprüft werden."
]

labels = ["kritisch", "unkritisch", "kritisch"]

# Texte und Labels zu Dokumenten verbinden
dokumente = [
    {
        "id": f"B{nummer:02d}",
        "text": text,
        "label": label
    }
    for nummer, (text, label) in enumerate(
        zip(texte, labels, strict=True),
        start=1
    )
]

def ist_verwertbar(dokument):
    # Komplexere Prüfung in einer Funktion bündeln
    pflichtfelder = ("id", "text", "label")
    return all(dokument.get(feld) for feld in pflichtfelder)

# Kritische Dokumente mit einer Listenkomprehension auswählen
kritische_dokumente = [
    dokument
    for dokument in dokumente
    if dokument["label"] == "kritisch"
]

# Prüfen, ob mindestens ein Dokument kritisch ist
hat_kritische_meldung = any(
    dokument["label"] == "kritisch"
    for dokument in dokumente
)

# Prüfen, ob alle Dokumente vollständig sind
alle_verwertbar = all(
    ist_verwertbar(dokument)
    for dokument in dokumente
)

print(f"Dokumente: {dokumente}")
print(f"Kritische Dokumente: {len(kritische_dokumente)}")
print(f"Kritische Meldung vorhanden: {hat_kritische_meldung}")
print(f"Alle Dokumente verwertbar: {alle_verwertbar}")
