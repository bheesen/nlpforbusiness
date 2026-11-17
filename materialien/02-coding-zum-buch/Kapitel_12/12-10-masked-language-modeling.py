# Ein BERT-ähnlicher Encoder rekonstruiert ein maskiertes Wort
#%% Verzeichnisse definieren
from pathlib import Path
import sys

BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
from nlp_eda import positionskodierung, setze_seed
import torch
import spacy
from spacy.symbols import ORTH
 
setze_seed()
# <mask> als ein Token erhalten.
nlp = spacy.blank("de")
nlp.tokenizer.add_special_case("<mask>", [{ORTH: "<mask>"}])

def tokenisiere(text):
    return [t.text for t in nlp.make_doc(text)]
paare = [
    ("sensor meldet <mask>", "fehler"),
    ("anzeige meldet <mask>", "fehler"),
    ("pumpe verliert <mask>", "öl"),
    ("leitung verliert <mask>", "kühlwasser"),
    ("lager ist <mask>", "verschlissen"),
    ("ventil ist <mask>", "undicht"),
]
woerter = sorted(
    {w for text, _ in paare for w in tokenisiere(text)} |
    {ziel for _, ziel in paare}
)
index = {w: i for i, w in enumerate(woerter)}
x = torch.tensor([
    [index[w] for w in tokenisiere(text)]
    for text, _ in paare
])
y = torch.tensor([index[ziel] for _, ziel in paare])
mask_id = index["<mask>"]
emb = torch.nn.Embedding(len(woerter), 16)
schicht = torch.nn.TransformerEncoderLayer(
    16, 4, 32, 0.0, batch_first=True
)
encoder = torch.nn.TransformerEncoder(schicht, 1)
kopf = torch.nn.Linear(16, len(woerter))
parameter = list(emb.parameters()) + list(encoder.parameters())
parameter += list(kopf.parameters())
opt = torch.optim.Adam(parameter, lr=0.03)
for _ in range(140):
    opt.zero_grad()
    h = emb(x) + positionskodierung(3, 16)
    verborgen = encoder(h)
    pos = x.eq(mask_id).nonzero()[:, 1]
    logits = kopf(verborgen[torch.arange(len(x)), pos])
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    opt.step()
with torch.no_grad():
    probs = logits[0].softmax(0)
werte, ids = probs.topk(3)
print("Eingabe: sensor meldet <mask>")
for wert, nr in zip(werte, ids):
    print(f"  {woerter[int(nr)]:<12} {float(wert):.3f}")
print("Trainingsverlust:", round(float(loss.detach()), 4)) 
