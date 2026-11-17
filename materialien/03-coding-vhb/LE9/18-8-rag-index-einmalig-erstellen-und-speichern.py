# Organisationswissen einmalig als persistenten RAG-Index vorbereiten.
#%% Verzeichnisse und Pakete
from pathlib import Path
import json
import faiss
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
DOKUMENTE = BASIS / "EigeneDaten" / "Organisation"
AUSGABE = BASIS / "EigenerOutput"
INDEX = AUSGABE / "organisation.faiss"
META = AUSGABE / "organisation-rag.json"
EMBEDDING = "intfloat/multilingual-e5-small"
GROESSE = 120
UEBERLAPPUNG = 25
def lese_text(pfad):
    # Häufige Dokumentformate in einfachen Text überführen.
    suffix = pfad.suffix.lower()
    if suffix in {".txt", ".md"}:
        return pfad.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(pfad)
        return "\n".join(
            seite.extract_text() or "" for seite in reader.pages
        )
    if suffix == ".docx":
        doc = Document(pfad)
        return "\n".join(absatz.text for absatz in doc.paragraphs)
    return ""
def teile(text):
    # Dokumente in überlappende Wortfenster zerlegen.
    woerter = text.split()
    schritt = GROESSE - UEBERLAPPUNG
    return [
        " ".join(woerter[i:i + GROESSE])
        for i in range(0, len(woerter), schritt)
        if woerter[i:i + GROESSE]
    ]
# Dokumente einlesen und Passagen mit Quellen speichern.
passagen = []
anzahl_dokumente = 0
formate = {".txt", ".md", ".pdf", ".docx"}
for pfad in sorted(DOKUMENTE.rglob("*")):
    if not pfad.is_file() or pfad.suffix.lower() not in formate:
        continue
    teile_dokument = teile(lese_text(pfad))
    if not teile_dokument:
        continue
    anzahl_dokumente += 1
    for text in teile_dokument:
        passagen.append({"quelle": pfad.name, "text": text})
if not passagen:
    raise RuntimeError(f"Keine Dokumente in {DOKUMENTE} gefunden.")
# Passagen einbetten und als FAISS-Index speichern.
encoder = SentenceTransformer(EMBEDDING)
matrix = encoder.encode(
    ["passage: " + p["text"] for p in passagen],
    normalize_embeddings=True,
    show_progress_bar=False,
).astype("float32")
index = faiss.IndexFlatIP(matrix.shape[1])
index.add(matrix)
AUSGABE.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(INDEX))
# Metadaten sichern die Zuordnung von Vektor, Text und Quelle.
daten = {
    "embedding_model": EMBEDDING,
    "chunk_size": GROESSE,
    "overlap": UEBERLAPPUNG,
    "passagen": passagen,
}
META.write_text(
    json.dumps(daten, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
# Umfang der Wissensbasis und Speicherdateien ausgeben.
print("\nERGEBNIS DER RAG-INDEXIERUNG")
print(f"{'Merkmal':24s} | Wert")
print("-" * 56)
print(f"{'Eingelesene Dokumente':24s} | {anzahl_dokumente}")
print(f"{'Erzeugte Passagen':24s} | {len(passagen)}")
print(f"{'FAISS-Index':24s} | {INDEX.name}")
print(f"{'Metadaten':24s} | {META.name}")

