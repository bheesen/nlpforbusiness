# spaCy- und scikit-learn-Tokenisierung vergleichen
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Leere deutsche spaCy-Pipeline für die Tokenisierung erzeugen.
nlp = spacy.blank("de")
text = "E-Mail an service@example.org senden!"


def spacy_tokenisiere(text):
    # spaCy tokenisiert; casefold() vereinheitlicht die Schreibweise.
    return [
        token.text.casefold()
        for token in nlp.make_doc(text)
        if not token.is_punct
    ]

# 1. Tokens direkt mit spaCy erzeugen.
spacy_tokens = spacy_tokenisiere(text)

# 2. Standard-Tokenisierung des CountVectorizer verwenden.
count = CountVectorizer()
count.fit([text])
count_tokens = count.get_feature_names_out().tolist()

# 3. Standard-Tokenisierung des TfidfVectorizer verwenden.
tfidf = TfidfVectorizer()
tfidf.fit([text])
tfidf_tokens = tfidf.get_feature_names_out().tolist()

# 4. spaCy als benutzerdefinierten TF-IDF-Tokenizer einsetzen.
tfidf_spacy = TfidfVectorizer(
    # spaCy übernimmt die Tokenisierung.
    tokenizer=spacy_tokenisiere,
    # Standard-RegEx von scikit-learn deaktivieren.
    token_pattern=None,
    # Keine zusätzliche Kleinschreibung durch scikit-learn,
    # da die Normalisierung bereits mit casefold() erfolgt.
    lowercase=False,
)
tfidf_spacy.fit([text])
tfidf_spacy_tokens = tfidf_spacy.get_feature_names_out().tolist()

print("spaCy:")
print(spacy_tokens)
print("\nCountVectorizer:")
print(count_tokens)
print("\nTfidfVectorizer:")
print(tfidf_tokens)
print("\nTF-IDF mit spaCy-Tokenizer:")
print(tfidf_spacy_tokens)
print("\nGleiche scikit-learn-Tokens:",
      count_tokens == tfidf_tokens)
