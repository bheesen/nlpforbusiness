# Konstituenten- und Abhängigkeitsparsing
import nltk
import spacy
# Eine kleine Grammatik für den Konstituentenparser definieren.
grammatik = nltk.CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP PP
PP -> P NP
Det -> 'Der' | 'den' | 'der'
N -> 'Techniker' | 'Motor' | 'Werkhalle'
V -> 'prüft'
P -> 'in'
""")
# Den ersten Satz mit NLTK hierarchisch analysieren.
satz = "Der Techniker prüft den Motor in der Werkhalle"
# Satz für den NLTK-Parser mit spaCy tokenisieren.
tokenizer = spacy.blank("de")
satz_tokens = [t.text for t in tokenizer.make_doc(satz)]
parser = nltk.ChartParser(grammatik)
baum = next(parser.parse(satz_tokens))
baum.pretty_print(unicodelines=True)
# Ein trainiertes deutsches spaCy-Modell laden.
nlp = spacy.load("de_core_news_sm")
# Zwei unterschiedliche Wortstellungen analysieren.
saetze = ["Der Techniker prüft den Motor in der Werkhalle.",
    "Den Motor prüft der Techniker in der Werkhalle.",]
# Abhängigkeitsrelationen und syntaktische Köpfe ausgeben.
for satz in saetze:
    print(f"\n{satz}")
    for token in nlp(satz):
        print(f"{token.text:12} {token.dep_:8} -> {token.head.text}")
# Erklärung der Abkürzungen von spaCy
print("\nErklärung der spaCy Abkürzungen:")
for label in ["sb", "oa", "nk", "mo", "punct", "ROOT"]:
    print(f"{label:<6} {spacy.explain(label)}")
    
    
