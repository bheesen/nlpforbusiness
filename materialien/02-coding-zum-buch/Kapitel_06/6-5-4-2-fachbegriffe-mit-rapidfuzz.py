# Leicht abweichende Fachbegriffe einem Referenzwert zuordnen.
from rapidfuzz import process

begriffe = [
    "Zentrifugalpumpe 4",
    "Kühlmittelpumpe 2",
    "Drucksensor 7",
]

eingabe = "Zentrifugalpume 4"
treffer, score, _ = process.extractOne(
    eingabe,
    begriffe,
)

print("Eingabe:", eingabe)
print("Treffer:", treffer)
print("Ähnlichkeit:", round(score, 1))

