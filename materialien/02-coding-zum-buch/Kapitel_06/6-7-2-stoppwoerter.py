# Negation als Stoppwort kann eine Klassifikation verschlechtern
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Trainingsbeispiele für die Klassifikation
train = ["gut", "sehr gut", "nicht schlecht", "schlecht",
         "sehr schlecht", "nicht gut"]
# Zuordnung 1=positiv, 0=negativ von labels zu train
labels = [1, 1, 1, 0, 0, 0]
# Neue Texte für die Vorhersage
test = ["gar nicht gut", "gar nicht schlecht"]
# ngram_range=(1, 2) legt fest, welche zusammenhängenden Wortfolgen der
# TfidfVectorizer als Merkmale verwendet. Die erste Zahl bezeichnet die
# kürzeste und die zweite Zahl die längste berücksichtigte Wortfolge:
# (1, 1) verwendet nur einzelne Wörter, sogenannte Unigramme.
# (2, 2) verwendet nur Wortpaare, sogenannte Bigramme.
# (1, 2) verwendet einzelne Wörter und Wortpaare
# Die logistische Regression übernimmt in der Pipeline die Klassifikation der
# zuvor durch TF-IDF numerisch dargestellten Texte. Beim Training ermittelt sie
# für jedes Wort und jedes Wortpaar ein Gewicht, das entweder stärker für die
# positive oder für die negative Klasse spricht. Für einen neuen Text werden
# die Gewichte der vorhandenen Merkmale zu einem Gesamtwert zusammengeführt und
# in eine Klassenwahrscheinlichkeit umgerechnet. Das Modell versteht Negationen
# dabei nicht aufgrund sprachlicher Regeln, sondern lernt aus den
# Trainingsbeispielen, dass Merkmale wie „nicht gut“ eher mit der negativen und
# „nicht schlecht“ eher mit der positiven Klasse verbunden sind. Werden die
# Negationswörter vor der Klassifikation entfernt, gehen diese entscheidenden
# Merkmale verloren und die Vorhersage kann sich umkehren. Das sehr kleine
# Trainingskorpus dient allerdings nur der Veranschaulichung und wäre für eine
# zuverlässige praktische Anwendung nicht ausreichend.
for stop in [None, ["gar", "nicht"]]:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), stop_words=stop),
        LogisticRegression(random_state=0),
    )
    model.fit(train, labels)
    print("Stoppwörter:", stop, "Vorhersage:", model.predict(test).tolist())
#%% Informationsverlust durch Stoppwortfilterung zeigen
import spacy
# Deutsche Sprachklasse mit Standard-Stoppwortliste laden.
nlp = spacy.blank("de")
texte = [
    "Die Pumpe läuft nicht ruhig.",
    "Die Pumpe läuft ruhig.",
]
print("STOPPWORTFILTERUNG")
for text in texte:
    doc = nlp(text)
    inhalt = [
        token.text
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    print(f"\nOriginal:   {text}")
    print(f"Gefiltert:  {' '.join(inhalt)}")
