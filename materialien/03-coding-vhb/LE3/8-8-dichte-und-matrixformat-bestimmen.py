# Speicherbedarf einer dünn besetzten Matrix untersuchen
from sklearn.feature_extraction.text import CountVectorizer

texte = [
    "Pumpe vibriert stark",
    "Motor wird warm",
    "Ventil klemmt",
    "Dichtung verliert Öl",
]

print("Texte:")
for nr, doc in enumerate(texte, start=1):
    print(f"Text-{nr}: {doc}")
matrix = CountVectorizer().fit_transform(texte)
zellen = matrix.shape[0] * matrix.shape[1]
dichte = matrix.nnz / zellen
print("\nForm:", matrix.shape)
print("Gespeicherte Werte:", matrix.nnz)
print("Alle Matrixzellen:", zellen)
print(f"Dichte: {dichte:.3f}")
print("Format:", type(matrix).__name__)
