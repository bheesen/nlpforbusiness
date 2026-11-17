# RAG-Chatbot mit frei wählbarem Antwortstil.
# Der FAISS-Index wird zuvor mit 22-8-rag-index-einmalig-erstellen-und-
# speichern.py erzeugt.
# Organisationswissen kommt aus RAG, der Nutzer bestimmt die Darstellungsform.
#%% Verzeichnisse und Pakete
from pathlib import Path
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
# Projekt- und Ausgabeverzeichnis bestimmen.
BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
AUSGABE = BASIS / "EigenerOutput"
# Persistenter FAISS-Index mit den Dokument-Embeddings.
INDEX = AUSGABE / "organisation.faiss"
# Metadaten enthalten Quellen, Textpassagen und Embedding-Modell.
META = AUSGABE / "organisation-rag.json"
# Qwen erzeugt aus Frage, RAG-Kontext und Stilwunsch die Antwort.
# Grobe Praxiswerte für FP16/BF16 ohne Quantisierung:
# bis etwa 4 GB VRAM: Qwen/Qwen3-0.6B
# etwa 6 bis 8 GB VRAM: Qwen/Qwen3-1.7B
# ab etwa 12 GB VRAM: Qwen/Qwen3-4B-Instruct-2507
# CPU-Offloading kann größere Modelle ermöglichen, ist aber meist langsamer.
LLM = "Qwen/Qwen3-0.6B"
# Der Encoder bleibt auf der CPU, damit mehr GPU-Speicher für Qwen frei ist.
ENCODER_GERAET = "cpu"
# Höchstens drei relevante Dokumentpassagen an das LLM übergeben.
TOP_K = 3
# Nur ausreichend ähnliche Treffer als Wissenskontext verwenden.
# Der Wert ist keine Wahrscheinlichkeit und muss praktisch validiert werden.
MIN_SCORE = 0.75
# Maximale Länge der neu generierten Antwort.
MAX_NEUE_TOKENS = 140
# Antwort bei fehlendem oder zu schwachem Dokumentkontext.
FALLBACK = (
    "Dazu liegen in den gefundenen Dokumenten keine verlässlichen "
    "Informationen vor."
)
#%% Vorbereiteten FAISS-Index und Metadaten einlesen
# Beide Dateien müssen durch das Index-Skript erzeugt worden sein.
if not INDEX.exists() or not META.exists():
    raise FileNotFoundError(
        "FAISS-Index oder Metadaten fehlen. "
        "Zuerst 22-8-rag-index-einmalig-erstellen-und-speichern.py ausführen."
    )
# Die Metadaten stellen die Verbindung zwischen FAISS-Vektor,
# Textpassage und ursprünglicher Quelldatei her.
daten = json.loads(META.read_text(encoding="utf-8"))
passagen = daten["passagen"]
# Der FAISS-Index enthält bereits alle Dokument-Embeddings.
# Sie werden beim Start nur geladen und nicht erneut berechnet.
index = faiss.read_index(str(INDEX))
# Index und Metadaten müssen dieselbe Anzahl von Passagen enthalten.
if index.ntotal != len(passagen):
    raise ValueError(
        "FAISS-Index und Metadaten enthalten verschieden viele Passagen."
    )
# Für Suchfragen muss dasselbe Embedding-Modell verwendet werden,
# mit dem zuvor die Dokumentpassagen indexiert wurden.
encoder = SentenceTransformer(
    daten["embedding_model"],
    device=ENCODER_GERAET,
)
# Auch die Vektordimension muss zum gespeicherten Index passen.
dimension = encoder.get_embedding_dimension()
if dimension != index.d:
    raise ValueError(
        "Embedding-Modell und FAISS-Index haben verschiedene Dimensionen."
    )
#%% Sprachmodell laden
# Der Tokenizer überführt die Chat-Nachrichten in Modell-Tokens.
tokenizer = AutoTokenizer.from_pretrained(LLM)
# device_map="auto" verteilt Qwen bei Bedarf auf GPU und CPU.
modell = AutoModelForCausalLM.from_pretrained(
    LLM,
    torch_dtype="auto",
    device_map="auto",
    offload_buffers=True,
)
modell.eval()
def suchen(frage, k=TOP_K):
    """Passende Organisationspassagen zur fachlichen Frage suchen."""
    # E5 verwendet für Suchanfragen das Präfix "query:".
    query = encoder.encode(
        ["query: " + frage],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    # Nicht mehr Treffer verlangen, als im Index vorhanden sind.
    k = min(k, index.ntotal)
    # FAISS liefert Ähnlichkeitswerte und IDs der besten Passagen.
    scores, ids = index.search(query, k)
    # Nur ausreichend ähnliche Treffer als Kontext zulassen.
    return [
        (
            passagen[i]["quelle"],
            passagen[i]["text"],
            float(score),
        )
        for score, i in zip(scores[0], ids[0])
        if i >= 0 and score >= MIN_SCORE
    ]
def generieren(nachrichten):
    """Aus RAG-Kontext und Stilwunsch eine Antwort erzeugen."""
    # Das Chat-Template erzeugt das von Qwen erwartete Eingabeformat.
    text = tokenizer.apply_chat_template(
        nachrichten,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Die Eingabetokens an das Eingabegerät des Modells übertragen.
    eingabe = tokenizer(
        text,
        return_tensors="pt",
    ).to(modell.device)
    # Bei der Inferenz werden keine Gradienten benötigt.
    with torch.inference_mode():
        ausgabe = modell.generate(
            **eingabe,
            max_new_tokens=MAX_NEUE_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Nur die neu erzeugten Tokens in Text zurückverwandeln.
    neu = ausgabe[0][eingabe["input_ids"].shape[1]:]
    return tokenizer.decode(
        neu,
        skip_special_tokens=True,
    ).strip()
def antworten(frage, stil):
    """Fakten per RAG suchen und im gewünschten Stil formulieren."""
    # Das Retrieval nutzt nur die fachliche Frage.
    # Der Stilwunsch darf die Dokumentensuche nicht beeinflussen.
    treffer = suchen(frage)
    # Ohne ausreichend gute Wissensquelle wird nicht geraten.
    if not treffer:
        return FALLBACK, "keine ausreichenden Treffer"
    # Relevante Passagen mit Quelle und Retrieval-Score zusammenstellen.
    kontext = "\n\n".join(
        f"[{quelle} | {score:.3f}]\n{text}"
        for quelle, text, score in treffer
    )
    # Fakten müssen aus RAG stammen.
    # Der Nutzer darf dagegen Form, Länge und Stil der Antwort bestimmen.
    systemtext = (
        "Du bist ein interner Serviceassistent. Beantworte "
        "organisationsspezifische Fragen ausschließlich mit Informationen "
        "aus dem bereitgestellten Kontext. Erfinde keine Fachbegriffe, "
        "Abkürzungen oder Anweisungen. Der gewünschte Antwortstil darf "
        "nur die Darstellung verändern, aber keine neuen Fakten erzeugen. "
        "Reicht der Kontext nicht aus, antworte: "
        f"'{FALLBACK}'"
    )
    # Kontext, Frage und frei formulierter Stilwunsch des Nutzers
    # werden gemeinsam an das Sprachmodell übergeben.
    nachrichten = [
        {
            "role": "system",
            "content": systemtext,
        },
        {
            "role": "user",
            "content": (
                f"KONTEXT:\n{kontext}\n\n"
                f"FRAGE:\n{frage}\n\n"
                f"GEWÜNSCHTER ANTWORTSTIL:\n{stil}"
            ),
        },
    ]
    # Qwen formuliert dieselben Fakten in der gewünschten Form.
    antwort = generieren(nachrichten)
    # Verwendete Quelldateien ohne Wiederholungen ausgeben.
    refs = ", ".join(
        dict.fromkeys(
            quelle
            for quelle, _, _ in treffer
        )
    )
    return antwort, refs
#%% Drei Antwortstile mit derselben fachlichen Frage testen
frage = "Was bedeutet VIB-H bei ZP4 und was ist zuerst zu prüfen?"
# Die Angaben simulieren frei formulierte Wünsche eines Chatbot-Nutzers.
stile = [
    (
        "Einfach",
        "Antworte einfach und in höchstens zwei kurzen Sätzen.",
    ),
    (
        "Strukturiert",
        "Antworte mit den Feldern Alarm, Anlage und Erste Maßnahme.",
    ),
    (
        "Arbeitsanweisung",
        "Antworte als kurze nummerierte Arbeitsanweisung im Imperativ.",
    ),
]
for name, stil in stile:
    antwort, refs = antworten(frage, stil)
    print(f"\n{name}")
    print("-" * len(name))
    print("Frage:", frage)
    print("Stilwunsch:", stil)
    print("Antwort:", antwort)
    print("Quellen:", refs)

