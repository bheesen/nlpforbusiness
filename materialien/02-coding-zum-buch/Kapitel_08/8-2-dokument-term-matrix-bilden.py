# Eine Dokument-Term-Matrix mit CountVectorizer bilden
from sklearn.feature_extraction.text import CountVectorizer

texte = [
    "Pumpe vibriert stark",
    "Motor wird warm",
    "Pumpe verliert Öl",
]

vektorisierer = CountVectorizer(lowercase=True)
matrix = vektorisierer.fit_transform(texte)
print("Texte:")
for nr, doc in enumerate(texte, start=1):
    print(f"Text-{nr}: {doc}")
print("\nMerkmale:")
print(vektorisierer.get_feature_names_out().tolist())
print("\nDokument-Term-Matrix:")
print(matrix.toarray())
print("\nForm:", matrix.shape)
