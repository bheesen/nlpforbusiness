# Daten reproduzierbar und stratifiziert aufteilen
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from collections import Counter
from nlp_eda import WARTUNGSDATEN_TEXTE, WARTUNGSDATEN_LABELS
from sklearn.model_selection import train_test_split

texte, labels = WARTUNGSDATEN_TEXTE, WARTUNGSDATEN_LABELS
# Texte und Labels gemeinsam in Training und Test aufteilen.
x_train, x_test, y_train, y_test = train_test_split(
    texte,
    labels,
    # 25 % der Dokumente für den späteren Test zurückhalten.
    test_size=0.25,
    # Fester Startwert macht die Zufallsaufteilung reproduzierbar.
    # Die konkrete Zahl 7 ist frei gewählt.
    random_state=7,
    # Klassenverteilung entsprechend labels in Training und Test beibehalten.
    stratify=labels,
)
# Anzahl der Dokumente in Training und Test ausgeben.
print(f"Training: {len(x_train)} Dokumente")
print(f"Test:     {len(x_test)} Dokumente")
# Counter zählt die Dokumente je Klasse.
# items() liefert Paare aus Klassenname und Anzahl.
# sorted() sortiert diese Paare alphabetisch nach Klassenname.
print(f"Training: {dict(sorted(Counter(y_train).items()))}")
print(f"Test:     {dict(sorted(Counter(y_test).items()))}")
# set() wandelt die Textlisten in Mengen um.
# Der Operator & bildet die Schnittmenge beider Mengen.
# Der Wert 0 zeigt, dass kein Text in beiden Teilmengen vorkommt.
print(f"Überschneidung: {len(set(x_train) & set(x_test))}")
