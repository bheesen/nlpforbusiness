# Wartungsberichte in strukturierte Datensätze überführen

import json
import spacy

# Deutsche Sprachverarbeitung mit Satzsegmentierung vorbereiten.
nlp = spacy.blank("de")
nlp.add_pipe("sentencizer")

berichte = [
    (
        "Pumpe 4: starke Vibrationen. Maßnahme: Lager prüfen. "
        "Frist: 14 Tage. Priorität: hoch."
    ),
    (
        "Ventil 2: Ölverlust. Maßnahme: Dichtung ersetzen. "
        "Frist: 3 Tage. Priorität: mittel."
    ),
]

def teile_satz(satz):
    # Bezeichnung und Inhalt am Doppelpunkt trennen.
    pos = next(i for i, t in enumerate(satz) if t.text == ":")
    links = satz[:pos].text.strip()
    rechts = satz[pos + 1:].text.strip(" .")
    return links, rechts

def analysiere(text):
    # Bericht in einzelne Informationssegmente zerlegen.
    saetze = list(nlp(text).sents)
    anlage, problem = teile_satz(saetze[0])
    daten = {"Anlage": anlage, "Problem": problem}
    # Die weiteren Feldbezeichnungen und Werte übernehmen.
    for satz in saetze[1:]:
        feld, wert = teile_satz(satz)
        daten[feld] = wert
    return daten

for text in berichte:
    daten = analysiere(text)
    print(json.dumps(daten, ensure_ascii=False, indent=2))
