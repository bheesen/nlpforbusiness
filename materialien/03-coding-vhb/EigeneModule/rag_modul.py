# Qwen und den FAISS-Index aus Kapitel 14 zusammenführen.
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pfade import modellordner, ragordner


class RagChatbot:
    def __init__(self):
        # E5 und den bestehenden Index mit Metadaten laden.
        self.encoder = SentenceTransformer(
            str(modellordner() / "e5"), local_files_only=True,
            device="cuda" if torch.cuda.is_available() else "cpu")
        root = ragordner()
        self.index = faiss.read_index(str(root / "wartung_rag.faiss"))
        datei = root / "wartung_rag.json"
        self.dokumente = json.loads(datei.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.dokumente):
            raise ValueError("Index und Metadaten passen nicht zusammen")
        if self.index.d != 384:
            raise ValueError("E5-small benötigt Vektoren mit 384 Dimensionen")
        pfad = modellordner() / "qwen"
        self.tok = AutoTokenizer.from_pretrained(
            pfad, local_files_only=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            pfad, local_files_only=True, dtype="auto", device_map="auto")
        self.llm.eval()
        # Sampling-Werte deaktivieren: feste, reproduzierbare Ausgabe.
        cfg = self.llm.generation_config
        cfg.do_sample = False
        cfg.temperature = 1.0
        cfg.top_p = 1.0
        cfg.top_k = 50

    def antwort(self, frage, k=1):
        # Präfix und Normalisierung entsprechen exakt Kapitel 14.
        if not frage.strip():
            raise ValueError("Bitte eine Frage eingeben")
        query = self.encoder.encode(
            ["query: " + frage], normalize_embeddings=True
        ).astype("float32")
        _, ids = self.index.search(query, min(k, self.index.ntotal))
        treffer = [self.dokumente[int(i)] for i in ids[0] if i >= 0]
        kontext = "\n".join(
            f"[{d['id']}] {d['titel']}: {d['text']}" for d in treffer)
        #%% System-Prompt für dokumentengestützte Antworten
        system = (
            "Beantworte die Frage auf Deutsch anhand des Kontexts. "
            "Gib die im passenden Dokument genannten Maßnahmen wieder. "
            "Erfinde keine Informationen. "
            "Nenne die ID des verwendeten Dokuments."
        )
        chats = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Kontext:\n{kontext}\nFrage: {frage}"},
        ]
        prompt = self.tok.apply_chat_template(
            chats, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self.tok(prompt, return_tensors="pt")
        inputs = {a: b.to(self.llm.device) for a, b in inputs.items()}
        with torch.inference_mode():
            # Verwendete Generierungseinstellungen anzeigen.
            print("Modellgerät:", self.llm.device)
            print("Prompt:", repr(prompt))
            print("Top-k:", k)
            print("Neue Tokens:", 180)
            output = self.llm.generate(
                **inputs, max_new_tokens=180, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        neu = output[0][inputs["input_ids"].shape[1]:]
        text = self.tok.decode(neu, skip_special_tokens=True)
        # Antwort und Originaldokumente zurückgeben.
        return {
            "antwort": text.strip(),
            "quellen": [d["id"] for d in treffer],
            "passagen": [
                f"[{d['id']}] {d['titel']}: {d['text']}"
                for d in treffer
            ],
        }
