# Die Mehrdeutigkeit eines statischen Wortvektors sichtbar machen
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import kosinus, satzvektor, svd_embedding, wortvektor

woerter, vektoren = svd_embedding(dimension=8)
statisch = wortvektor("lager", woerter, vektoren)
mechanik = satzvektor("lager defekt welle vibriert", woerter, vektoren)
logistik = satzvektor("ersatzteile material im lager", woerter, vektoren)
print(f"Statischer Lagervektor zu Mechanik: {kosinus(statisch, mechanik):.3f}")
print(f"Statischer Lagervektor zu Logistik: {kosinus(statisch, logistik):.3f}")
print(f"Mechanik zu Logistik:               {kosinus(mechanik, logistik):.3f}")
print("Ein statischer Vektor mischt beide Bedeutungen.")
