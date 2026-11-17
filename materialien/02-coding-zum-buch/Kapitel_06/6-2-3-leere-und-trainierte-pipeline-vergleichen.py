# Leere und trainierte Pipeline vergleichen
import spacy
from importlib.metadata import version

# Leere und trainierte deutsche Pipeline erzeugen.
leer = spacy.blank("de")
modell = "de_core_news_sm"
trainiert = spacy.load(modell)

# Versionen für die Reproduzierbarkeit bestimmen.
spacy_version = spacy.__version__
modell_version = version(modell)

print("SOFTWARE UND MODELL")
print(f"spaCy-Version:  {spacy_version}")
print(f"Pipeline-Paket: {modell}")
print(f"Modellversion:  {modell_version}")

print("\nLEERE PIPELINE")
print("Komponenten: keine")

print("\nTRAINIERTE PIPELINE")
for nummer, name in enumerate(trainiert.pipe_names, 1):
    print(f"{nummer:>2}. {name}")
