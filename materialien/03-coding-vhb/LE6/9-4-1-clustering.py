# Wartungsmeldungen mit TF-IDF und K-Means clustern.
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
texte = [
    "Pumpe vibriert stark, Lager der Pumpe ist laut.",
    "Pumpe zeigt starke Vibrationen am Lager.",
    "Lager der Pumpe zeigt starke Vibrationen.",
    "Pumpe mit Vibrationen und unruhigem Lager.",
    "Motor heiß, Temperatur steigt am Motor.",
    "Hohe Temperatur am Motor, Kühlung prüfen.",
    "Motor überhitzt, Temperatur am Motor zu hoch.",
    "Kühlung schwach, Motor wird heiß und Temperatur steigt.",
    "Sensor liefert falsche Messwerte.",
    "Sensor meldet fehlerhafte Messwerte.",
    "Sensor zeigt falsche Position und Messwerte.",
    "Sensorwerte unplausibel, Sensor prüfen.",
]
# Texte in TF-IDF-Merkmale überführen.
vektor = TfidfVectorizer(ngram_range=(1, 2))
matrix = vektor.fit_transform(texte)
# Drei Gruppen ohne vorgegebene Klassenlabels bilden.
modell = KMeans(n_clusters=3, random_state=7, n_init=20)
labels = modell.fit_predict(matrix)
# Dokumente gruppiert ausgeben.
for nummer in sorted(set(labels)):
    print(f"\nCluster {nummer}")
    for text, label in zip(texte, labels):
        if label == nummer:
            print("-", text)