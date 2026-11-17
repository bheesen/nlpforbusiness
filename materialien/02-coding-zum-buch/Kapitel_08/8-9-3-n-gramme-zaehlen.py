from collections import Counter
import re

text = """
Das Modell lernt aus Daten.
Das Modell erkennt Muster in Daten.
Das Modell lernt aus Beispielen.
"""

# Text zunächst in Sätze zerlegen
sentences = re.split(r"[.!?]+", text)
bigrams = []
trigrams = []
for sentence in sentences:
    words = re.findall(r"\b\w+\b", sentence.lower())
    # Bigramme
    bigrams.extend(tuple(words[i:i+2])
                   for i in range(len(words) - 1))
    # Trigramme
    trigrams.extend(tuple(words[i:i+3])
                    for i in range(len(words) - 2))

# Häufigkeiten zählen
bigram_counts = Counter(bigrams)
trigram_counts = Counter(trigrams)
print("Text:", text)
print("BI-Gramme:")
for ngram, frequency in bigram_counts.most_common():
    print(f'{" ".join(ngram):30} {frequency}')
print("\nTRI-Gramme:")
for ngram, frequency in trigram_counts.most_common():
    print(f'{" ".join(ngram):30} {frequency}')
