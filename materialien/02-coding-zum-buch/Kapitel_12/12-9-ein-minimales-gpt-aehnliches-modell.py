# Ein kleiner Decoder-only-Transformer lernt das nächste Wort
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

setze_seed()
# spaCy für die explizite Worttokenisierung verwenden.
nlp = spacy.blank("de")


def tokenisiere(text):
    return [t.text for t in nlp.make_doc(text) if t.is_alpha]

saetze = [
    "pumpe verliert öl",
    "ventil verliert druck",
    "sensor meldet fehler",
    "anzeige meldet fehler",
]
woerter = ["<pad>", "<bos>", "<eos>"] + sorted(
    {w for s in saetze for w in tokenisiere(s)}
)
index = {w: i for i, w in enumerate(woerter)}
folgen = [
    [index["<bos>"]] + [index[w] for w in tokenisiere(s)]
    for s in saetze
]

x = torch.tensor(folgen)
y = torch.tensor([f[1:] + [index["<eos>"]] for f in folgen])
emb = torch.nn.Embedding(len(woerter), 16)
schicht = torch.nn.TransformerEncoderLayer(
    16, 4, 32, 0.0, batch_first=True
)
encoder = torch.nn.TransformerEncoder(schicht, 1)
kopf = torch.nn.Linear(16, len(woerter))
parameter = list(emb.parameters()) + list(encoder.parameters())
parameter += list(kopf.parameters())
opt = torch.optim.Adam(parameter, lr=0.03)
maske = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
for _ in range(120):
    opt.zero_grad()
    h = emb(x) + positionskodierung(4, 16)
    logits = kopf(encoder(h, mask=maske))
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, len(woerter)), y.reshape(-1)
    )
    loss.backward()
    opt.step()
prompt = torch.tensor([[index["<bos>"], index["pumpe"], index["verliert"]]])
with torch.no_grad():
    h = emb(prompt) + positionskodierung(3, 16)
    lm = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    probs = kopf(encoder(h, mask=lm))[0, -1].softmax(0)
werte, ids = probs.topk(3)
print("Prompt: <bos> pumpe verliert")
for wert, nr in zip(werte, ids):
    print(f"  {woerter[int(nr)]:<8} {float(wert):.3f}")
print("Trainingsverlust:", round(float(loss.detach()), 4)) 
