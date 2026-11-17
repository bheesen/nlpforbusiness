# Dokumente mit Ähnlichkeit vergleichen
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

texte = [
    "Pumpe vibriert stark",
    "Starke Vibration an der Pumpe",
    "Motor wird heiß",
]

print("Texte:")
for nr, doc in enumerate(texte, start=1):
    print(f"Text-{nr}: {doc}")
# Texte in TF-IDF-Vektoren umwandeln
matrix = TfidfVectorizer().fit_transform(texte)
# Cosinus-Ähnlichkeit berechnen
aehnlichkeit = cosine_similarity(matrix)
# Euklidische Distanz berechnen
distanz = euclidean_distances(matrix)

print("\nCosinus-Ähnlichkeitsmatrix:")
print(aehnlichkeit.round(3))
print("\nEuklidische Distanzmatrix:")
print(distanz.round(3))
print("\nVergleich mit Text-1:")
for nr in range(len(texte)):
    print(
        f"Text-1 zu Text-{nr + 1}: "
        f"Cosinus = {aehnlichkeit[0, nr]:.3f}, "
        f"Distanz = {distanz[0, nr]:.3f}"
    )
