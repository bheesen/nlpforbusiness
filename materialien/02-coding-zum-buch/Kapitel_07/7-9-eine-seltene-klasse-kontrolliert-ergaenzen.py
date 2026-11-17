# Seltene Klassen mit kontrollierten Vorlagen ergänzen
import pandas as pd

korpus = pd.DataFrame({
    "text": ["Pumpe ist laut.", "Motor ist warm.", "Ventil klemmt."],
    "klasse": ["akustisch", "temperatur", "mechanisch"],
})
vorlagen = {
    "mechanisch": ["Ventil blockiert.", "Schieber bewegt sich nicht."],
}
neu = [
    {"text": text, "klasse": klasse, "synthetisch": True}
    for klasse, texte in vorlagen.items()
    for text in texte
]
korpus["synthetisch"] = False
erweitert = pd.concat([korpus, pd.DataFrame(neu)], ignore_index=True)
print(erweitert.to_string(index=False))
print("\nKlassenverteilung:")
print(erweitert["klasse"].value_counts())
