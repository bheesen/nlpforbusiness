# Ein vortrainiertes LLM bewertet mögliche nächste Tokens.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# CUDA-GPU verwenden, sonst CPU nutzen
geraet = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Rechengerät:", geraet)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
# Kleines Qwen-Modell für lokale Experimente
MODELL = "Qwen/Qwen3-0.6B"
text = "Die Pumpe verliert"
# Tokenizer und vortrainiertes Modell laden
tokenizer = AutoTokenizer.from_pretrained(MODELL)
modell = AutoModelForCausalLM.from_pretrained(
    MODELL,
    dtype=(
        torch.float16
        if geraet.type == "cuda"
        else torch.float32
    )
)
modell.to(geraet)
modell.eval()
# Text in Token-IDs umwandeln
eingabe = tokenizer(
    text,
    return_tensors="pt"
).to(geraet)
# Modell berechnet Wahrscheinlichkeiten für nächste Tokens
with torch.no_grad():
    logits = modell(**eingabe).logits[0, -1]
    probs = logits.softmax(dim=0)
# Die fünf wahrscheinlichsten Tokens bestimmen
werte, ids = probs.topk(5)
print("Kontext:", text)
print("Fünf wahrscheinlichste Folgetokens:")
for wert, token_id in zip(werte, ids):
    token = tokenizer.decode([int(token_id)])
    print(f"  {token!r:<16} {float(wert):.3f}")

