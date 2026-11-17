# Klassen und Wahrscheinlichkeiten mit logistischer Regression schätzen
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
# Texte und zugehörige Klassenlabels aus dem eigenen Modul laden.
from nlp_eda import WARTUNGSDATEN_TEXTE, WARTUNGSDATEN_LABELS
# Funktion zur Aufteilung in Trainings- und Testdaten importieren.
from sklearn.model_selection import train_test_split
# Kennzahlen zur Bewertung der Klassifikation importieren.
from sklearn.metrics import accuracy_score, classification_report
# TF-IDF wandelt Texte in gewichtete numerische Merkmale um.
from sklearn.feature_extraction.text import TfidfVectorizer
# Logistische Regression als Klassifikationsverfahren importieren.
from sklearn.linear_model import LogisticRegression
# Verarbeitungsschritte zu einer gemeinsamen Pipeline verbinden.
from sklearn.pipeline import make_pipeline

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

# Pipeline aus TF-IDF-Vektorisierung und Klassifikator erzeugen.
modell = make_pipeline(
    # Unigramme und Bigramme als Merkmale berücksichtigen.
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(
        # Maximale Zahl der Iterationen bei der Optimierung.
        max_iter=1000,
        # Fester Startwert unterstützt reproduzierbare Ergebnisse.
        random_state=7,
    ),
)
# Vektorisierer und Klassifikator nur mit Trainingsdaten lernen.
modell.fit(x_train, y_train)
#%% Vorhersage 5 Testdatensätze
# Fünf Dokumente aus dem Testdatensatz für eine Demo auswählen.
demo_texte = x_test[:5]
demo_labels = y_test[:5]
# Für die Beispiele Wahrscheinlichkeiten aller Klassen bestimmen.
demo_probs = modell.predict_proba(demo_texte)
# Klassennamen in der vom Modell verwendeten Reihenfolge abrufen.
klassen = modell.classes_
# Überschriften für die Demo-Ausgabe erzeugen.
print("BEISPIELE AUS DEM TESTDATENSATZ")
print("-" * 75)
print(
    f"{'Vorhersage':<12} "
    f"{'Real':<12} "
    f"{'Wahrsch.':>9}  "
    f"Text"
)
print("-" * 75)
# Vorhersage, korrekte Klasse und höchsten Wert ausgeben.
for text, korrekt, werte in zip(
    demo_texte,
    demo_labels,
    demo_probs,
):
    # argmax() liefert den Index der höchsten Wahrscheinlichkeit.
    index = werte.argmax()
    print(
        f"{klassen[index]:<12} "
        f"{korrekt:<12} "
        f"{werte[index]:>9.3f}  "
        f"{text}"
    )
#%% Vorhersage Testdaten
# Klassen für alle Testdaten vorhersagen.
prognose = modell.predict(x_test)
# Accuracy gibt den Anteil korrekt klassifizierter Dokumente an.
print(
    f"Accuracy: "
    f"{accuracy_score(y_test, prognose):.3f}"
)
# Precision, Recall und F1-Score für jede Klasse ausgeben.
# digits=3 legt drei Nachkommastellen für die Ausgabe fest.
print(
    classification_report(
        y_test,
        prognose,
        digits=3,
    )
)
