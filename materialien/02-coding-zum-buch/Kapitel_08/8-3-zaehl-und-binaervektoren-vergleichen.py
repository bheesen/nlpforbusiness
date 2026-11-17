# Häufigkeits- und Binärvektoren vergleichen
from sklearn.feature_extraction.text import CountVectorizer

texte = [
    "Pumpe Pumpe vibriert",
    "Pumpe verliert Öl",
]

zaehlung = CountVectorizer()
binaer = CountVectorizer(binary=True)
mat_z = zaehlung.fit_transform(texte)
mat_b = binaer.fit_transform(texte)
print("Merkmale:", zaehlung.get_feature_names_out().tolist())
print("Zählwerte:")
print(mat_z.toarray())
print("\nBinärwerte:")
print(mat_b.toarray())
