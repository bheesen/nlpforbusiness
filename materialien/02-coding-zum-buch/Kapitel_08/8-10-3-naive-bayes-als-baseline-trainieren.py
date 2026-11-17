# Einen Multinomial-Naive-Bayes-Klassifikator trainieren
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
# Multinomial Naive Bayes als Klassifikationsverfahren importieren.
from sklearn.naive_bayes import MultinomialNB
# Mehrere Verarbeitungsschritte zu einer Pipeline verbinden.
from sklearn.pipeline import make_pipeline
# Texte über Worthäufigkeiten in numerische Merkmale umwandeln.
from sklearn.feature_extraction.text import CountVectorizer

texte, labels = WARTUNGSDATEN_TEXTE, WARTUNGSDATEN_LABELS

# Texte und Labels gemeinsam in Trainings- und Testdaten aufteilen.
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

# Pipeline aus Vektorisierung und Klassifikator erzeugen.
# CountVectorizer zählt, wie oft Wörter in jedem Dokument vorkommen.
# MultinomialNB nutzt diese Häufigkeiten für die Klassifikation.
modell = make_pipeline(
    CountVectorizer(),
    MultinomialNB(),
)

# fit() lernt zuerst das Vokabular und danach das Klassifikationsmodell.
# Beide Schritte verwenden ausschließlich die Trainingsdaten.
modell.fit(x_train, y_train)

# predict() sagt für jedes bisher nicht verwendete Testdokument
# eine der im Training gelernten Klassen voraus.
prognose = modell.predict(x_test)

# Accuracy gibt den Anteil korrekt klassifizierter Dokumente an.
print(
    f"Accuracy: "
    f"{accuracy_score(y_test, prognose):.3f}"
)

# classification_report zeigt Precision, Recall und F1 je Klasse.
# digits=3 legt drei Nachkommastellen für die Ausgabe fest.
print(
    classification_report(
        y_test,
        prognose,
        digits=3,
    )
)
