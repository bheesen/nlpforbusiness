# Texte in eine dünn besetzte TF-IDF-Matrix überführen.
from sklearn.feature_extraction.text import TfidfVectorizer

texte = ["Pumpe zeigt starke Vibrationen",
         "Sensor meldet hohe Temperatur",
         "Pumpe meldet hohe Temperatur",]
vektorisierer = TfidfVectorizer()
matrix = vektorisierer.fit_transform(texte)
# Speicherbedarf von Sparse- und dichter Matrix vergleichen.
dicht = matrix.toarray()
speicher_dicht = dicht.nbytes
speicher_sparse = (
    matrix.data.nbytes
    + matrix.indices.nbytes
    + matrix.indptr.nbytes
)
ersparnis = 100 * (1 - speicher_sparse / speicher_dicht)
print(f"Typ: {type(matrix).__name__}")
print(f"Form: {matrix.shape}")
print(f"Nichtnullwerte: {matrix.nnz}")
print(f"Dicht: {speicher_dicht} Bytes")
print(f"Sparse: {speicher_sparse} Bytes")
print(f"Ersparnis: {ersparnis:.1f} %")
print("Merkmale:",  vektorisierer.get_feature_names_out().tolist())
print(dicht.round(2))

