# TF-IDF mit Mindesthäufigkeit und sublinearer TF anpassen
from sklearn.feature_extraction.text import TfidfVectorizer

texte = [
    "Pumpe Pumpe Pumpe vibriert",
    "Pumpe verliert Öl",
    "Motor wird warm",
    "Motor vibriert leicht",
]

vektorisierer = TfidfVectorizer(
    min_df=2,
    sublinear_tf=True,
)
matrix = vektorisierer.fit_transform(texte)
print("Verbleibende Merkmale:")
print(vektorisierer.get_feature_names_out().tolist())
print("\nTF-IDF-Matrix:")
print(matrix.toarray().round(3))
