# Wartungsmeldungen mit TF-IDF und K-Means clustern.
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
# Unterschiedliche Clusterzahlen vergleichen.
from sklearn.metrics import silhouette_score
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

for k in range(2, 6):
    modell = KMeans(n_clusters=k, random_state=7, n_init=20)
    labels = modell.fit_predict(matrix)
    wert = silhouette_score(matrix, labels, metric="cosine")
    print(f"k={k}: Silhouette={wert:.3f}")


