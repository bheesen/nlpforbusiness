# RAG-Chatbot mit Einstellungen aus einer YAML-Datei.
from pathlib import Path
import json
import faiss
import torch
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
#%% Pfade und Konfiguration
SCRIPT = Path(__file__).resolve()
BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
CONFIG = EigeneDaten / "18-9-2-rag-config.yaml"
# YAML-Konfiguration aus dem Datenverzeichnis einlesen.
with CONFIG.open(encoding="utf-8") as datei:
    config = yaml.safe_load(datei)
llm = config["llm"]
rag = config["rag"]
INDEX = EigenerOutput / rag["index"]
META = EigenerOutput / rag["metadaten"]
#%% Gespeicherten RAG-Index laden
if not INDEX.exists() or not META.exists():
    raise FileNotFoundError(
        "RAG-Dateien fehlen. Zuerst Kapitel 22.8 ausführen."
    )
daten = json.loads(META.read_text(encoding="utf-8"))
passagen = daten["passagen"]
index = faiss.read_index(str(INDEX))
if index.ntotal != len(passagen):
    raise ValueError("Index und Metadaten passen nicht zusammen.")
encoder = SentenceTransformer(
    daten["embedding_model"],
    device=rag["encoder_geraet"],
)
if encoder.get_embedding_dimension() != index.d:
    raise ValueError("Embedding-Modell und Index passen nicht zusammen.")
#%% Sprachmodell laden
tokenizer = AutoTokenizer.from_pretrained(llm["modell"])
modell = AutoModelForCausalLM.from_pretrained(
    llm["modell"],
    torch_dtype="auto",
    device_map="auto",
    offload_buffers=True,
)
modell.eval()
#%% Retrieval und Generierung
def suchen(frage):
    # Die aktuelle Frage als E5-Suchanfrage einbetten.
    query = encoder.encode(
        ["query: " + frage],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    k = min(rag["top_k"], index.ntotal)
    scores, ids = index.search(query, k)
    return [
        (
            passagen[i]["quelle"],
            passagen[i]["text"],
            float(score),
        )
        for score, i in zip(scores[0], ids[0])
        if i >= 0 and score >= rag["min_score"]
    ]
def generieren(nachrichten):
    # Chat-Nachrichten in Modell-Tokens umwandeln.
    text = tokenizer.apply_chat_template(
        nachrichten,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    eingabe = tokenizer(
        text,
        return_tensors="pt",
    ).to(modell.device)
    with torch.inference_mode():
        ausgabe = modell.generate(
            **eingabe,
            max_new_tokens=llm["max_neue_tokens"],
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    start = eingabe["input_ids"].shape[1]
    return tokenizer.decode(
        ausgabe[0][start:],
        skip_special_tokens=True,
    ).strip()
def antworten(frage, verlauf):
    # Letzte Nutzerfrage verbessert das Retrieval bei Folgefragen.
    vorige = [
        n["content"]
        for n in verlauf
        if n["role"] == "user"
    ]
    suchfrage = " ".join(vorige[-1:] + [frage])
    treffer = suchen(suchfrage)
    if not treffer:
        return rag["fallback"], "keine ausreichenden Treffer"

    kontext = "\n\n".join(
        f"[{q} | {s:.3f}]\n{t}"
        for q, t, s in treffer
    )
    system = (
        "Du bist ein interner Serviceassistent. Antworte nur mit "
        "Informationen aus dem bereitgestellten Kontext. Erfinde "
        "keine internen Begriffe oder Anweisungen."
    )
    nachrichten = [{"role": "system", "content": system}]
    nachrichten.extend(verlauf[-4:])
    nachrichten.append(
        {
            "role": "user",
            "content": f"KONTEXT:\n{kontext}\n\nFRAGE:\n{frage}",
        }
    )
    antwort = generieren(nachrichten)
    refs = ", ".join(dict.fromkeys(q for q, _, _ in treffer))
    return antwort, refs
#%% Mehrturnigen Chatbot testen
fragen = [
    "Was bedeutet VIB-H bei ZP4?",
    "Welches Lager soll ich dann prüfen?",
    "Welcher Materialcode gehört zu L2?",
]
verlauf = []
for frage in fragen:
    antwort, refs = antworten(frage, verlauf)
    print("\nFrage:", frage)
    print("Antwort:", antwort)
    print("Quellen:", refs)
    verlauf += [
        {"role": "user", "content": frage},
        {"role": "assistant", "content": antwort},
    ]
