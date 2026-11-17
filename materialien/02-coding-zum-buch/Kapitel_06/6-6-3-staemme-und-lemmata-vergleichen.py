#%% Lemmatisieren
import spacy
from nltk.stem.snowball import GermanStemmer
# Trainierte deutsche Pipeline laden.
nlp = spacy.load("de_core_news_sm")
# Stemmer
stemmer = GermanStemmer()
text = "prüft geprüft Prüfungen prüfte Lager Lagern"
doc = nlp(text)
print("Lemmatisierung und Stemming")
print(f"{'Token':<12} {'Lemma':<12} {'Stamm':<12}")
print("-" * 36)
# Für das Beispiel nur Nomen und Verben ausgeben.
for token in doc:
    stem = stemmer.stem(token.text)
    if token.pos_ in {"NOUN", "VERB"}:
        print(f"{token.text:<12} {token.lemma_:<12} {stem:<12} ")
        
        
