# Übereinstimmung zweier Annotierender mit Cohen-Kappa messen
from sklearn.metrics import cohen_kappa_score, confusion_matrix

labels_a = ["hoch", "hoch", "mittel", "niedrig", "mittel", "hoch"]
labels_b = ["hoch", "mittel", "mittel", "niedrig", "niedrig", "hoch"]
klassen = ["niedrig", "mittel", "hoch"]

kappa = cohen_kappa_score(labels_a, labels_b, labels=klassen)
matrix = confusion_matrix(labels_a, labels_b, labels=klassen)
print(f"Cohen-Kappa: {kappa:.3f}")
print("\nKonfusionsmatrix:")
print("             B: niedrig mittel hoch")
for name, zeile in zip(klassen, matrix):
    werte = " ".join(f"{wert:6d}" for wert in zeile)
    print(f"A: {name:7} {werte}")
