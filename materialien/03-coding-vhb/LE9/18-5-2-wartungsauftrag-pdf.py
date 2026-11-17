# Vorhandenes PDF-Formular mit extrahierten Angaben befuellen.
#%% Verzeichnisse definieren
from pathlib import Path
import sys
import os
BASIS = Path(
    r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP"
)
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS
#%% Pakete importieren
import spacy
from pypdf import PdfReader, PdfWriter
from spacy.matcher import Matcher
# Organisationsspezifische Fachbegriffe erkennen.
nlp = spacy.load("de_core_news_md")
ruler = nlp.add_pipe(
    "entity_ruler",
    after="ner",
    config={"overwrite_ents": True},
)
ruler.add_patterns([
    {"label": "ANLAGE", "pattern": "ZP4"},
    {
        "label": "ALARM",
        "pattern": [{"TEXT": {"REGEX": r"(?i)^VIB-H\.?$"}}],
    },
    {"label": "BAUTEIL", "pattern": "L2"},
    {"label": "SENSOR", "pattern": "VS-22"},
])
# Weitere Felder mit tokenbasierten Mustern ermitteln.
matcher = Matcher(nlp.vocab)
matcher.add("MESSWERT", [[
    {"LIKE_NUM": True}, {"LOWER": "mm"}, {"ORTH": "/"},
    {"LOWER": {"REGEX": r"^s\.?$"}},
]])
matcher.add("MASSNAHME", [[
    {"ENT_TYPE": "BAUTEIL"}, {"LOWER": "zu"}, {"LOWER": "prüfen"},
]])
matcher.add("PRIORITAET", [[
    {"LOWER": "priorität"}, {"IS_PUNCT": True, "OP": "?"},
    {"LOWER": {"IN": ["hoch", "mittel", "niedrig"]}},
]])
# Freien Wartungsbericht analysieren.
text = (
    "ZP4 meldet VIB-H. VS-22 zeigt 8,2 mm/s. "
    "Vor dem Wiederanlauf ist L2 zu prüfen. Priorität: hoch."
)
print("AUSGANGSTEXT DER WARTUNGSMELDUNG")
print(text)
doc = nlp(text)
auftrag = {
    ent.label_.lower(): ent.text.rstrip(".,")
    for ent in doc.ents
    if ent.label_ in {"ANLAGE", "ALARM", "BAUTEIL", "SENSOR"}
}
for match_id, start, ende in matcher(doc):
    name = nlp.vocab.strings[match_id]
    span = doc[start:ende]
    if name == "MESSWERT":
        auftrag["messwert"] = span.text.rstrip(".,")
    elif name == "MASSNAHME":
        auftrag["massnahme"] = f"{span[0].text} prüfen"
    elif name == "PRIORITAET":
        auftrag["prioritaet"] = span[-1].text.lower()
# Der Status ist eine Geschäftsregel, keine extrahierte Information.
auftrag["status"] = "offen"
auftrag["originaltext"] = text
# Pflichtfelder vor dem Ausfuellen des Formulars pruefen.
pflicht = [
    "anlage", "alarm", "sensor", "messwert", "bauteil",
    "massnahme", "prioritaet",
]
fehlend = [feld for feld in pflicht if not auftrag.get(feld)]
if fehlend:
    raise ValueError("Pflichtfelder fehlen: " + ", ".join(fehlend))
# Extrahierte Auftragsdaten mit Spaltenüberschriften ausgeben.
print("\nERKANNTE AUFTRAGSDATEN")
print(f"{'Merkmal':15s} | Wert")
print("-" * 55)
for merkmal, wert in auftrag.items():
    if merkmal != "originaltext":
        print(f"{merkmal:15s} | {wert}")
#%% Vorhandenes PDF-Formular befuellen und speichern
vorlage = EigeneDaten / "Wartungsauftrag_Formular.pdf"
ziel = EigenerOutput / "wartungsauftrag_zp4_ausgefuellt.pdf"
EigenerOutput.mkdir(parents=True, exist_ok=True)
leser = PdfReader(vorlage)
felder = set(leser.get_fields() or {})
werte = {**auftrag, "freigabe_durch": ""}
fehlende_felder = set(werte) - felder
if fehlende_felder:
    raise ValueError(
        "PDF-Felder fehlen: " + ", ".join(sorted(fehlende_felder))
    )
schreiber = PdfWriter()
schreiber.append(leser)
# Freigabe bleibt leer und muss durch einen Menschen erfolgen.
schreiber.update_page_form_field_values(
    schreiber.pages[0],
    werte,
    auto_regenerate=True,
)
with ziel.open("wb") as datei:
    schreiber.write(datei)
print(f"\nPDF-Wartungsauftrag gespeichert unter:\n{ziel.resolve()}")
# Gespeicherte PDF-Datei im Windows-PDF-Viewer anzeigen.
os.startfile(str(ziel.resolve()))

