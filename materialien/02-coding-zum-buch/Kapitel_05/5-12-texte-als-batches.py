# Große Textmengen als Generator in Batches verarbeiten
from collections.abc import Iterable, Iterator

def bereinigte_texte(texte: Iterable[str]) -> Iterator[str]:
    for text in texte:
        text = text.strip()
        if text:
            yield text

def batches(texte: Iterable[str], groesse: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for text in bereinigte_texte(texte):
        batch.append(text)
        if len(batch) == groesse:
            yield batch
            batch = []
    if batch:
        yield batch

texte = ["Pumpe prüfen", " ", "Sensor tauschen", "Motor stoppen", "Reinigen"]
for nummer, batch in enumerate(batches(texte, 2), start=1):
    print(f"Batch {nummer}: {batch}")
