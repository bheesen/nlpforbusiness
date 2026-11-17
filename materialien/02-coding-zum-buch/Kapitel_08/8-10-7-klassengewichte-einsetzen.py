# Den Einfluss von Klassenungleichgewicht untersuchen
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

normale = [
    "Routineprüfung ohne dringenden Befund",
    "Anlage läuft nach der Wartung stabil",
    "Leichte Geräusche weiter beobachten",
    "Kleine Abweichung bei der Sichtprüfung",
] * 12
kritische = [
    "Anlage sofort stoppen wegen Rauch",
    "Akute Leckage erfordert sofortige Abschaltung",
    "Kabelbrand macht sofortiges Eingreifen nötig",
    "Schutzhaube gebrochen Anlage sofort sperren",
    "Starker Druckverlust erfordert Notabschaltung",
    "Motor überhitzt Anlage unmittelbar stoppen",
]
texte = normale + kritische
labels = ["normal"] * len(normale) + ["kritisch"] * len(kritische)
x_train, x_test, y_train, y_test = train_test_split(
    texte,
    labels,
    test_size=0.35,
    random_state=3,
    stratify=labels,
)
for gewicht in [None, "balanced"]:
    modell = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(
            class_weight=gewicht,
            max_iter=1000,
            random_state=7,
        ),
    )
    modell.fit(x_train, y_train)
    prognose = modell.predict(x_test)
    recall = recall_score(y_test, prognose, pos_label="kritisch")
    print(f"class_weight={str(gewicht):<8} Recall kritisch={recall:.3f} "
          f"Prognosen kritisch={sum(prognose == 'kritisch')}")
