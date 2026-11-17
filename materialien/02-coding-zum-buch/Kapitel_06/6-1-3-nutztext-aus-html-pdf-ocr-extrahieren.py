# Nutztext aus HTML, PDF und gescanntem Dokument extrahieren.

#%% Verzeichnisse definieren
from pathlib import Path
import sys
BASIS = Path(r"C:\Users\bernd\Documents\A-Python\EigenerKurs\WorkshopNLP")
EigeneDaten = BASIS / "EigeneDaten"
EigenerOutput = BASIS / "EigenerOutput"
sys.path.insert(0, str(BASIS / "EigeneModule"))
del BASIS

#%% Pakete importieren
import pdfplumber
import pytesseract
from bs4 import BeautifulSoup
from PIL import Image

#%% HTML-Boilerplate entfernen
html = (
    "<html><body><nav>Start | Kontakt</nav>"
    "<h1>Störung</h1>"
    "<p>Pumpe 4 vibriert stark.</p>"
    "<footer>Impressum</footer></body></html>"
)
soup = BeautifulSoup(html, "html.parser")
for tag in soup.select("nav, footer, script, style"):
    tag.decompose()

html_parts = [
    tag.get_text(" ", strip=True)
    for tag in soup.select("h1, p")
]
html_text = "\n".join(html_parts)

#%% Text aus digitalem PDF extrahieren
pdf_pfad = EigeneDaten / "wartungsbericht.pdf"
with pdfplumber.open(pdf_pfad) as pdf:
    pdf_text = "\n".join(
        seite.extract_text() or ""
        for seite in pdf.pages
    ).strip()

#%% OCR auf gescanntes Dokument anwenden
sprachen = pytesseract.get_languages(config="")
print(f"Deutsche OCR verfügbar: {'deu' in sprachen}")

if "deu" not in sprachen:
    raise RuntimeError("Tesseract-Sprachdaten 'deu' fehlen.")

scan_pfad = EigeneDaten / "wartungsbericht_scan.png"
ocr_text = pytesseract.image_to_string(
    Image.open(scan_pfad),
    lang="deu",
).strip()

#%% Ergebnisse ausgeben
print("\nHTML:")
print(html_text)
print("\nPDF:")
print(pdf_text)
print("\nOCR:")
print(ocr_text)