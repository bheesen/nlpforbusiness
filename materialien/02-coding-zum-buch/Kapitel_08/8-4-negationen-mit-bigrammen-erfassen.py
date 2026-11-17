# Wort-N-Gramme zur Erfassung lokaler Reihenfolge erzeugen
from sklearn.feature_extraction.text import CountVectorizer

texte = [
    "Lager ist nicht defekt",
    "Lager ist defekt",
    "Pumpe läuft nicht ruhig",
]

# Unigramme und Bigramme berücksichtigen
vektorisierer = CountVectorizer(ngram_range=(1, 2))
# Vokabular lernen und Texte in Vektoren umwandeln
matrix = vektorisierer.fit_transform(texte)
# Alle Merkmale in der Reihenfolge der Vektorspalten abrufen
merkmale = vektorisierer.get_feature_names_out()

print("Alle Merkmale:")
print(list(merkmale))
print("\nVollständige Vektoren:")
for text, vektor in zip(texte, matrix.toarray()):
    print(f"{text}:")
    print(vektor)
# Nur Merkmale betrachten, die "nicht" enthalten
indizes = [
    index
    for index, merkmal in enumerate(merkmale)
    if "nicht" in merkmal
]
nicht_merkmale = [merkmale[index] for index in indizes]
print("\nBetrachtete Merkmale mit 'nicht':")
print(nicht_merkmale)
print("\nVerkürzte Vektoren:")
for text, vektor in zip(texte, matrix.toarray()):
    teilvektor = vektor[indizes]
    print(f"{text}:")
    print(teilvektor)
