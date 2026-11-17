# Vektorisierung und Modellparameter gemeinsam abstimmen
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
from nlp_eda import WARTUNGSDATEN_LABELS, WARTUNGSDATEN_TEXTE
# TF-IDF wandelt Texte in gewichtete numerische Merkmale um.
from sklearn.feature_extraction.text import TfidfVectorizer
# Funktionen für Kreuzvalidierung und Random Search importieren.
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
# Verarbeitungsschritte zu einer Pipeline verbinden.
from sklearn.pipeline import Pipeline
# Logistische Regression als Klassifikationsverfahren importieren.
from sklearn.linear_model import LogisticRegression

#%% Trainings- und Testdaten erzeugen
texte, labels = WARTUNGSDATEN_TEXTE, WARTUNGSDATEN_LABELS

# 25 % der Dokumente für den unabhängigen Test zurückhalten.
x_train, x_test, y_train, y_test = train_test_split(
    texte,
    labels,
    test_size=0.25,
    # Fester Startwert macht die Aufteilung reproduzierbar.
    random_state=7,
    # Klassenanteile in Training und Test erhalten.
    stratify=labels,
)

#%% Pipeline definieren
# TF-IDF-Vektorisierung und Klassifikator gemeinsam verarbeiten.
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(),
    ),
    (
        "modell",
        LogisticRegression(
            max_iter=1000,
            random_state=7,
        ),
    ),
])

#%% Parameter für die zufällige Suche festlegen
# Mögliche Werte für Vektorisierer und Modell definieren.
parameter = {
    # Nur Unigramme oder zusätzlich Bigramme berücksichtigen.
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2),
    ],
    # Optional logarithmisch skalierte Termfrequenz verwenden.
    "tfidf__sublinear_tf": [
        False,
        True,
    ],
    # Verschiedene Regularisierungswerte untersuchen.
    "modell__C": [
        0.5,
        1.0,
        2.0,
    ],
}

#%% Kreuzvalidierung definieren
# Vier stratifizierte Folds für jede Parameterkombination verwenden.
cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=7,
)

#%% Random Search durchführen
# RandomizedSearchCV prüft nur eine zufällige Auswahl der
# möglichen Parameterkombinationen.
suche = RandomizedSearchCV(
    pipeline,
    parameter,
    # Sechs zufällige Parameterkombinationen untersuchen.
    n_iter=6,
    cv=cv,
    # Macro-F1 gewichtet jede Klasse gleich stark.
    scoring="f1_macro",
    # Fester Startwert macht die Parameterauswahl reproduzierbar.
    random_state=7,
    # Berechnung bewusst auf einen Prozess beschränken.
    n_jobs=1,
)

# Hyperparameter ausschließlich mit den Trainingsdaten auswählen.
suche.fit(x_train, y_train)


#%% Ergebnis der Random Search ausgeben

print(f"Bester CV-F1-Makro-Wert: {suche.best_score_:.3f}")
print("Beste Parameter:")
print("-" * 40)
for name, wert in sorted(suche.best_params_.items()):
    print(f"{name:<24} {wert}")

#%% Unabhängigen Testdatensatz bewerten
# best_estimator_ ist die beste gefundene Pipeline.
# Sie wurde automatisch erneut auf allen Trainingsdaten angepasst.
bestes_modell = suche.best_estimator_
# Erst jetzt den bislang unangetasteten Testdatensatz verwenden.
prognose = bestes_modell.predict(x_test)
print("\nTESTBEWERTUNG")
print("-" * 60)
# Anteil korrekt klassifizierter Testdokumente ausgeben.
print(
    f"Test-Accuracy: "
    f"{accuracy_score(y_test, prognose):.3f}"
)
# Precision, Recall, F1-Score und Support je Klasse ausgeben.
print(
    classification_report(
        y_test,
        prognose,
        digits=3,
        zero_division=0,
    )
)
