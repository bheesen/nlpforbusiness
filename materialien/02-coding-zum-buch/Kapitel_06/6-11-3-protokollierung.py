# Konfiguration und Kennzahlen einer Pipeline protokollieren
import json
# Konfiguration und Versionen einer spaCy-Pipeline protokollieren
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
import spacy

@dataclass(frozen=True)
class Config:
    # Einstellungen und Regelstand festhalten.
    lowercase: bool = True
    min_length: int = 3
    regelstand: str = "2026-08-11"
    modell_id: str = "de_core_news_sm"

def process(texts, config, nlp):
    # Dieselbe Pipeline auf alle Texte anwenden.
    result = []
    for text in texts:
        if config.lowercase:
            text = text.casefold()
        doc = nlp(text)
        tokens = [t.text for t in doc if not t.is_punct]
        result.append(" ".join(tokens))

    metrics = {
        "dokumente": len(texts),
        "zu_kurz": sum(
            len(text) < config.min_length for text in texts
        ),
    }
    return result, metrics

config = Config()
nlp = spacy.load(config.modell_id)
eingaben = ["Pumpe 4!", "", "Lager prüfen."]
texte, metrics = process(eingaben, config, nlp)
versionen = {
    "python": platform.python_version(),
    "spacy": spacy.__version__,
    "modell": config.modell_id,
    "modell_version": version(config.modell_id),
}
protokoll = {
    "konfiguration": asdict(config),
    "versionen": versionen,
    "pipeline": nlp.pipe_names,
    "kennzahlen": metrics,
}
print("Texte:", texte)
print(json.dumps(protokoll, ensure_ascii=False, indent=2))
