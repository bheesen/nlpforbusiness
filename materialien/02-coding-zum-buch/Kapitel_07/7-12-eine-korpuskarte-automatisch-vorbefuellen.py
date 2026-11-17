# Eine kompakte Korpuskarte aus Kennzahlen erzeugen
import pandas as pd

korpus = pd.DataFrame({
    "quelle": ["Werk A", "Werk A", "Werk B", "Werk B"],
    "klasse": ["Pumpe", "Motor", "Pumpe", "Ventil"],
    "text": ["laut", "warm", "undicht", "klemmt"],
})
karte = f"""# Korpuskarte: Wartungsberichte

- Zweck: Demonstration von NLP-Verfahren
- Dokumente: {len(korpus)}
- Quellen: {korpus['quelle'].nunique()}
- Klassen: {', '.join(sorted(korpus['klasse'].unique()))}
- Sprache: Deutsch
- Einschränkung: kleines, nicht repräsentatives Lehrkorpus
"""
print(karte.strip())
