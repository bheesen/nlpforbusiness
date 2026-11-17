# Einen Vektor für ein unbekanntes Kompositum zusammensetzen
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import kosinus, svd_embedding, wortvektor
import numpy as np

def ngramme(wort):
    erweitert = f"<{wort}>"
    return {
        erweitert[i:i + n]
        for n in range(3, 6)
        for i in range(len(erweitert) - n + 1)
    }

woerter, basis = svd_embedding(dimension=6)
ngram_vektoren = {}
for wort in woerter:
    for gram in ngramme(wort):
        wert = wortvektor(wort, woerter, basis)
        ngram_vektoren.setdefault(gram, []).append(wert)
ngram_vektoren = {
    gram: np.mean(werte, axis=0)
    for gram, werte in ngram_vektoren.items()
}
neu = "pumpenlager"
teile = [ngram_vektoren[g] for g in ngramme(neu) if g in ngram_vektoren]
vektor = np.mean(teile, axis=0)
print(f"Bekanntes Wort: {neu in woerter}")
print(f"Nutzbare N-Gramme: {len(teile)}")
for wort in ["pumpe", "lager", "sensor", "dichtung"]:
    wert = kosinus(vektor, wortvektor(wort, woerter, basis))
    print(f"Ähnlichkeit zu {wort:<9}: {wert:.3f}")
