# Eine lineare Support Vector Machine für Texte einsetzen
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
import numpy as np
# Texte und zugehörige Klassenlabels aus dem eigenen Modul laden.
from nlp_eda import WARTUNGSDATEN_TEXTE, WARTUNGSDATEN_LABELS
# Funktion zur Aufteilung in Trainings- und Testdaten importieren.
from sklearn.model_selection import train_test_split
# Kennzahlen zur Bewertung der Klassifikation importieren.
from sklearn.metrics import accuracy_score, classification_report
# TF-IDF wandelt Texte in gewichtete numerische Merkmale um.
from sklearn.feature_extraction.text import TfidfVectorizer
# Verarbeitungsschritte zu einer gemeinsamen Pipeline verbinden.
from sklearn.pipeline import make_pipeline
# Lineare SVM zur Trennung der Dokumentklassen
from sklearn.svm import LinearSVC

texte, labels = WARTUNGSDATEN_TEXTE, WARTUNGSDATEN_LABELS

# Daten in Trainings- und Testdaten aufteilen.
x_train, x_test, y_train, y_test = train_test_split(
    texte,
    labels,
    # 25 % der Dokumente für den abschließenden Test reservieren.
    test_size=0.25,
    # Fester Zufallsstartwert macht die Aufteilung reproduzierbar.
    random_state=7,
    # Klassenanteile in Training und Test möglichst erhalten.
    stratify=labels,
)

# Pipeline aus TF-IDF-Vektorisierung und linearer SVM erzeugen.
modell = make_pipeline(
    # Unigramme und Bigramme als Textmerkmale berücksichtigen.
    TfidfVectorizer(ngram_range=(1, 2)),
    # Lineare Support Vector Machine zur Textklassifikation verwenden.
    # Sie sucht Trennflächen mit möglichst großem Abstand
    # zwischen den Klassen im hochdimensionalen Merkmalsraum.
    LinearSVC(
        random_state=7,
    ),
)

# Vektorisierer und SVM ausschließlich mit Trainingsdaten lernen.
modell.fit(x_train, y_train)

#%% Vorhersage 5 Testdatensätze
# Fünf Dokumente aus dem Testdatensatz für die Demo auswählen.
demo_texte = x_test[:5]
demo_labels = y_test[:5]
# Entscheidungswerte für alle Klassen berechnen.
demo_werte = modell.decision_function(demo_texte)
# Klassennamen der SVM abrufen.
klassen = modell.classes_
print("BEISPIELE AUS DEM TESTDATENSATZ")
print("-" * 75)
print(
    f"{'Vorhersage':<12} "
    f"{'Korrekt':<12} "
    f"{'Abstand':>9}  "
    f"Text"
)
print("-" * 75)
# Fünf Testdokumente mit Vorhersage und Soll-Klasse ausgeben.
for text, korrekt, werte in zip(
    demo_texte,
    demo_labels,
    demo_werte,
):
    # Klassen nach ihren Entscheidungswerten absteigend sortieren.
    rang = np.argsort(werte)[::-1]

    # Klasse mit dem höchsten Entscheidungswert auswählen.
    vorhersage = klassen[rang[0]]

    # Abstand zwischen bestem und zweitbestem Wert bestimmen.
    abstand = werte[rang[0]] - werte[rang[1]]

    print(
        f"{vorhersage:<12} "
        f"{korrekt:<12} "
        f"{abstand:>9.3f}  "
        f"{text}"
    )

#%% Vorhersage Testdaten
# Anschließend alle Dokumente des Testdatensatzes klassifizieren.
prognose = modell.predict(x_test)
print("\nGESAMTBEWERTUNG")
print("-" * 60)
# Accuracy gibt den Anteil korrekt klassifizierter Dokumente an.
print(
    f"Accuracy: "
    f"{accuracy_score(y_test, prognose):.3f}"
)
# Precision, Recall, F1-Score und Support je Klasse ausgeben.
# digits=3 legt drei Nachkommastellen für die Ausgabe fest.
print(
    classification_report(
        y_test,
        prognose,
        digits=3,
    )
)
