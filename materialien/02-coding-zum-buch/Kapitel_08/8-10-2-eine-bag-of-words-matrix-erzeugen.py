# Texte in eine Bag-of-Words-Matrix umwandeln
from sklearn.feature_extraction.text import CountVectorizer

texte = [
    "Pumpe vibriert stark",
    "Pumpe verliert Öl",
    "Sensor liefert falsche Werte",
    "Motor vibriert und wird heiß",
]
vektorisierer = CountVectorizer(lowercase=True)
matrix = vektorisierer.fit_transform(texte)
merkmale = vektorisierer.get_feature_names_out()
print(f"Form der Matrix: {matrix.shape}")
print("Merkmale:", ", ".join(merkmale))
print("Dokument-Term-Matrix:")
print(matrix.toarray())
