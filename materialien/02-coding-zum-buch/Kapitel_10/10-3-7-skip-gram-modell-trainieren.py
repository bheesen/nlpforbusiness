# Ein kleines Skip-Gram-Modell mit Softmax trainieren
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import kontextpaare, nachbarn
import numpy as np

# Zielwort-Kontextwort-Paare mit einem Fenster von zwei erzeugen.
woerter, paare = kontextpaare(fenster=2)
# Größe des Vokabulars und Dimension der Wortvektoren festlegen.
anzahl = len(woerter)
# Acht Dimensionen genügen für das kleine Lehrkorpus.
# In realen Anwendungen ist die Vektordimension ein Hyperparameter.
dimension = 8
# Zufallsgenerator für reproduzierbare Startwerte erzeugen.
rng = np.random.default_rng(7)
# Für jedes Wort einen kleinen zufälligen Startvektor erzeugen.
# Mittelwert 0 und Standardabweichung 0,1 halten die
# Anfangswerte nahe bei null.
eingang = rng.normal(
    0,
    0.1,
    (anzahl, dimension),
)
# Für jedes mögliche Kontextwort einen Ausgabevektor erzeugen.
ausgang = rng.normal(
    0,
    0.1,
    (anzahl, dimension),
)
# Schrittweite des Gradientenabstiegs festlegen.
lernrate = 0.025
# Das gesamte Trainingsmaterial 801-mal durchlaufen.
# Eine Epoche umfasst jeweils alle Zielwort-Kontextwort-Paare.
for epoche in range(801):
    verlust = 0.0
    # Ein Zielwort und sein beobachtetes Kontextwort trainieren.
    for zielwort, kontextwort in paare:
        # Vektor des aktuellen Zielworts auswählen.
        h = eingang[zielwort].copy()
        # Alte Ausgabematrix für das spätere Update sichern.
        alt = ausgang.copy()
        # Score für jedes mögliche Kontextwort berechnen.
        logits = ausgang @ h
        # Scores mit Softmax in Wahrscheinlichkeiten umwandeln.
        # Das Maximum wird zur numerischen Stabilisierung abgezogen.
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        # Verlust für das tatsächlich beobachtete Kontextwort.
        verlust += -np.log(probs[kontextwort] + 1e-12)
        # Gradient des Verlusts bezüglich der Scores: p - y.
        fehler = probs.copy()
        fehler[kontextwort] -= 1
        # Ausgabevektoren mit Gradientenabstieg aktualisieren.
        ausgang -= lernrate * np.outer(fehler, h)
        # Auch den Eingabevektor des Zielworts aktualisieren.
        eingang[zielwort] -= lernrate * (alt.T @ fehler)
    # Mittleren Verlust ausgewählter Epochen ausgeben.
    if epoche in [0, 200, 400, 800]:
        mittel = verlust / len(paare)
        print(f"Epoche {epoche:>3}: Verlust={mittel:.3f}")

# Ein- und Ausgabevektoren zu einer Repräsentation kombinieren.
modell = eingang + ausgang

print("")
# Je Wort die drei ähnlichsten Wortvektoren ausgeben.
for wort in ["pumpe", "sensor", "dichtung"]:
    print(
        wort,
        nachbarn(
            wort,
            woerter,
            modell,
            anzahl=3,
        ),
    )

