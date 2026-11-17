# Qwen3 erhält internes Wissen nur über bereitgestellten Kontext
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# CUDA-GPU verwenden, sonst CPU nutzen
geraet = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Rechengerät:", geraet)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
modellname = (
    "Qwen/Qwen3-1.7B"
    if geraet.type == "cuda" else "Qwen/Qwen3-0.6B"
)
dtype = torch.float16 if geraet.type == "cuda" else torch.float32
print("Modell:", modellname)
tokenizer = AutoTokenizer.from_pretrained(modellname)
modell = AutoModelForCausalLM.from_pretrained(
    modellname, torch_dtype=dtype
).to(geraet)
frage = "Was bedeutet XZ-17 bei Anlage P4?"
kontext = ("Handbuch: XZ-17 bezeichnet bei Anlage P4 den Prüfschritt "
           "für die Kühlmitteldichtung.")
def qwen3(prompt):
    chat = [{"role": "user", "content": prompt}]
    daten = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
        enable_thinking=False,
    ).to(geraet)
    with torch.inference_mode():
        tokens = modell.generate(
            **daten, max_new_tokens=50, do_sample=False
        )
    start = daten["input_ids"].shape[-1]
    return tokenizer.decode(
        tokens[0][start:], skip_special_tokens=True
    ).strip()

ohne = qwen3(frage)
mit = qwen3(f"Nutze nur den Kontext. Sonst: NICHT BELEGT.\n"
    f"Kontext: {kontext}\nFrage: {frage}")
print("Ohne Kontext:", ohne)
print("Mit Kontext:", mit)

