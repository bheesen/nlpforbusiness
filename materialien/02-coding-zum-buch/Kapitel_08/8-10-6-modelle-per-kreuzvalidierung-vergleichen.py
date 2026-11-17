# Mehrere Textklassifikatoren mit Kreuzvalidierung vergleichen
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
# TF-IDF wandelt Texte in gewichtete numerische Merkmale um.
from sklearn.feature_extraction.text import TfidfVectorizer
# Kennzahlen und Scorer zur Modellbewertung importieren.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
)
# Funktionen für Datenteilung und Kreuzvalidierung importieren.
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
# Verarbeitungsschritte zu einer Pipeline verbinden.
from sklearn.pipeline import make_pipeline
# Multinomial Naive Bayes importieren.
from sklearn.naive_bayes import MultinomialNB
# Logistische Regression als Klassifikationsverfahren importieren.
from sklearn.linear_model import LogisticRegression
# Lineare Support Vector Machine importieren.
from sklearn.svm import LinearSVC
# Baumbasierte Ensemble-Modelle importieren.
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)

#%% Trainings- und Testdaten erzeugen
texte, labels = WARTUNGSDATEN_TEXTE, WARTUNGSDATEN_LABELS

# 25 % der Dokumente als unabhängigen Testdatensatz zurückhalten.
x_train, x_test, y_train, y_test = train_test_split(
    texte,
    labels,
    test_size=0.25,
    # Fester Startwert macht die Aufteilung reproduzierbar.
    random_state=7,
    # Klassenanteile in Training und Test beibehalten.
    stratify=labels,
)

print(f"Training: {len(x_train)} Dokumente")
print(f"Test:     {len(x_test)} Dokumente")

#%% Kreuzvalidierung und Modelle definieren

# Trainingsdaten in vier stratifizierte Teilmengen aufteilen.
# Jede Teilmenge dient einmal zur Validierung.
cv = StratifiedKFold(
    n_splits=4,
    # Daten vor der Aufteilung zufällig mischen.
    shuffle=True,
    # Fester Startwert macht auch die Folds reproduzierbar.
    random_state=7,
)

# Macro-F1 gewichtet jede Klasse gleich stark.
# zero_division=0 behandelt fehlende Vorhersagen ohne Warnung.
f1_macro = make_scorer(
    f1_score,
    average="macro",
    zero_division=0,
)

# Jedes Modell erhält einen eigenen TF-IDF-Vektorisierer.
# Dieser wird in jedem CV-Durchlauf nur am Trainings-Fold angepasst.
modelle = {
    "Naive Bayes": make_pipeline(
        TfidfVectorizer(),
        MultinomialNB(),
    ),
    "Log. Regression": make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(
            max_iter=1000,
            random_state=7,
        ),
    ),
    "Lineare SVM": make_pipeline(
        TfidfVectorizer(),
        LinearSVC(
            random_state=7,
        ),
    ),
    "Random Forest": make_pipeline(
        TfidfVectorizer(),
        RandomForestClassifier(
            # Anzahl der nacheinander aufgebauten Bäume.
            n_estimators=200,
            max_features="sqrt",
            random_state=7,
        ),
    ),
    "Gradient Boosting": make_pipeline(
        TfidfVectorizer(),
        GradientBoostingClassifier(
            # Anzahl der nacheinander aufgebauten Bäume.
            n_estimators=100,
            # Bestimmt, wie stark jeder neue Baum das Gesamtmodell verändert.
            # 0.1 bedeutet: Jeder Baum trägt nur gedämpft zur Korrektur bei.
            learning_rate=0.1,
            # Kleine Bäume mit max. Tiefe als schwache Einzelmodelle verwenden.
            max_depth=3,
            random_state=7,
        ),
    ),
}

#%% Kreuzvalidierung und Training auf den Trainingsdaten
# Ergebnisse für den abschließenden Vergleich sammeln.
ergebnisse = []
for name, modell in modelle.items():
    print(f"\n{name.upper()}")
    print("-" * 60)
    # Vier F1-Macro-Werte durch Kreuzvalidierung bestimmen.
    cv_werte = cross_val_score(
        modell,
        x_train,
        y_train,
        cv=cv,
        scoring=f1_macro,
    )
    # Mittelwert beschreibt die durchschnittliche CV-Leistung.
    cv_mittel = cv_werte.mean()
    # Standardabweichung beschreibt die Streuung der vier Ergebnisse.
    cv_streuung = cv_werte.std()
    print(
        "CV-F1-Makro: "
        f"{cv_mittel:.3f} "
        f"(Streuung: {cv_streuung:.3f})"
    )
    # Modell danach mit allen Trainingsdaten neu trainieren.
    modell.fit(x_train, y_train)
    
#%% Vorhersage
    # Alle bislang zurückgehaltenen Testdokumente klassifizieren.
    prognose = modell.predict(x_test)

#%% Bewertung
    # Accuracy für Training und unabhängigen Test bestimmen.
    train_acc = modell.score(x_train, y_train)
    test_acc = accuracy_score(y_test, prognose)
    # Macro-F1 zusätzlich auf dem Testdatensatz berechnen.
    test_f1 = f1_score(
        y_test,
        prognose,
        average="macro",
        zero_division=0,
    )
    print(f"Training-Accuracy: {train_acc:.3f}")
    print(f"Test-Accuracy:     {test_acc:.3f}")
    print(f"Test-F1-Makro:     {test_f1:.3f}")
    # Precision, Recall, F1-Score und Support je Klasse ausgeben.
    print(
        classification_report(
            y_test,
            prognose,
            digits=3,
            zero_division=0,
        )
    )
    # Kennzahlen für die gemeinsame Tabelle speichern.
    ergebnisse.append(
        (
            name,
            cv_mittel,
            cv_streuung,
            train_acc,
            test_acc,
            test_f1,
        )
    )

#%% Alle fünf Modelle in einer Tabelle vergleichen

print("\nMODELLVERGLEICH")
print("-" * 75)
print(
    f"{'Modell':<19} "
    f"{'CV-F1':>8} "
    f"{'Streu.':>8} "
    f"{'Train':>8} "
    f"{'Test':>8} "
    f"{'Test-F1':>9}"
)
print("-" * 75)
for ergebnis in ergebnisse:
    name, cv_f1, streuung, train_acc, test_acc, test_f1 = ergebnis
    print(
        f"{name:<19} "
        f"{cv_f1:>8.3f} "
        f"{streuung:>8.3f} "
        f"{train_acc:>8.3f} "
        f"{test_acc:>8.3f} "
        f"{test_f1:>9.3f}"
    )
print("-" * 75)
