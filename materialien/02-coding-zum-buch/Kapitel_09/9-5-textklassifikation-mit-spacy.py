# Eine kleine Textklassifikation direkt mit spaCy trainieren.
import spacy
from spacy.training import Example
from spacy.util import fix_random_seed, minibatch

fix_random_seed(7)
nlp = spacy.blank("de")
textcat = nlp.add_pipe("textcat")
for label in ("mechanik", "temperatur"):
    textcat.add_label(label)

training = [
    ("Pumpe vibriert stark", "mechanik"),
    ("Lager macht laute Geräusche", "mechanik"),
    ("Welle läuft unruhig", "mechanik"),
    ("Motorlager rattert", "mechanik"),
    ("Motor wird sehr heiß", "temperatur"),
    ("Temperatur steigt schnell", "temperatur"),
    ("Kühlung ist ausgefallen", "temperatur"),
    ("Gehäuse überhitzt", "temperatur"),
]

def beispiele():
    # Pro Dokument genau eine Klasse als korrekt markieren.
    daten = []
    for text, label in training:
        cats = {name: float(name == label) for name in textcat.labels}
        daten.append(
            Example.from_dict(nlp.make_doc(text), {"cats": cats})
        )
    return daten

optimizer = nlp.initialize(get_examples=beispiele)
for _ in range(20):
    losses = {}
    for batch in minibatch(beispiele(), size=4):
        nlp.update(batch, sgd=optimizer, losses=losses, drop=0.1)

for text in ("Pumpe läuft unruhig", "Motor wird heiß"):
    doc = nlp(text)
    label = max(doc.cats, key=doc.cats.get)
    print(text)
    print("  Klasse:", label)
    print(
        "  Scores:",
        {k: round(v, 3) for k, v in doc.cats.items()},
    )
