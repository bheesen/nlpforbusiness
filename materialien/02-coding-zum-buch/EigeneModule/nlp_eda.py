# Hilfsfunktionen und synthetische Daten

import spacy
import torch
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Leere deutsche spaCy-Pipeline für die Tokenisierung erzeugen.
nlp = spacy.blank("de")

#%% Hilfsfunktionen

def tokenisiere(text):
    # Text mit dem deutschen spaCy-Tokenizer in Tokens zerlegen.
    doc = nlp.make_doc(text)
    # Nur Wörter und Ziffern auswählen und kleingeschrieben zurückgeben.
    return [
        token.lower_
        for token in doc
        if token.is_alpha or token.is_digit
    ]

def suche(anfrage, k=2):
    # Titel und Text zu Suchtexten verbinden.
    texte = [
        d["titel"] + " " + d["text"]
        for d in DOKUMENTE
    ]
    # TF-IDF mit derselben spaCy-Tokenisierung konfigurieren.
    vektor = TfidfVectorizer(
        tokenizer=tokenisiere,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
    )
    # Texte und Anfrage gemeinsam in TF-IDF-Vektoren umwandeln.
    matrix = vektor.fit_transform(texte + [anfrage])
    # Kosinusähnlichkeit zwischen Anfrage und Handbuchtexten berechnen.
    werte = cosine_similarity(
        matrix[-1],
        matrix[:-1],
    ).ravel()
    # Indizes der k ähnlichsten Dokumente absteigend sortieren.
    rang = werte.argsort()[::-1][:k]
    # Dokumente zusammen mit ihren Ähnlichkeitswerten zurückgeben.
    return [
        (DOKUMENTE[i], float(werte[i]))
        for i in rang
    ]

#%% Klassifikationsdaten für klassische NLP-Beispiele

# Jede Klasse enthält gleich viele kurze Wartungsberichte.
WARTUNGSDATEN = [
    ("Pumpe vibriert stark am Lager", "mechanik"),
    ("Motorlager erzeugt mahlende Geräusche", "mechanik"),
    ("Welle schlägt bei hoher Drehzahl", "mechanik"),
    ("Getriebe rattert unter Last", "mechanik"),
    ("Kupplung ist verschlissen und rutscht", "mechanik"),
    ("Riemen läuft schief und quietscht", "mechanik"),
    ("Lager der Pumpe wird ungewöhnlich heiß", "mechanik"),
    ("Antrieb zeigt starke Unwucht", "mechanik"),
    ("Welle hat zu großes Spiel", "mechanik"),
    ("Getriebezahn ist beschädigt", "mechanik"),
    ("Motor brummt und blockiert kurz", "mechanik"),
    ("Pumpe klappert beim Anfahren", "mechanik"),
    ("Riemenspannung ist deutlich zu niedrig", "mechanik"),
    ("Lager muss nachgeschmiert werden", "mechanik"),
    ("Kupplung schlägt beim Start", "mechanik"),
    ("Antrieb dreht ungleichmäßig", "mechanik"),
    ("Schaltschrank riecht verschmort", "elektrik"),
    ("Sicherung löst beim Start aus", "elektrik"),
    ("Motorstrom ist deutlich zu hoch", "elektrik"),
    ("Kabelisolierung ist beschädigt", "elektrik"),
    ("Klemme im Schaltschrank ist locker", "elektrik"),
    ("Frequenzumrichter meldet Überspannung", "elektrik"),
    ("Steuerung verliert kurz die Versorgung", "elektrik"),
    ("Relais schaltet nicht zuverlässig", "elektrik"),
    ("Leitung zeigt einen Kurzschluss", "elektrik"),
    ("Motor nimmt keinen Strom auf", "elektrik"),
    ("Schutzschalter fällt wiederholt ab", "elektrik"),
    ("Kabel am Motor ist stark erwärmt", "elektrik"),
    ("Spannung an Klemme drei fehlt", "elektrik"),
    ("Stecker hat einen Wackelkontakt", "elektrik"),
    ("Netzteil liefert instabile Spannung", "elektrik"),
    ("Kontakt im Relais ist verbrannt", "elektrik"),
    ("Pumpe verliert Öl an der Dichtung", "fluid"),
    ("Ventil tropft nach dem Schließen", "fluid"),
    ("Hydraulikleitung ist undicht", "fluid"),
    ("Kühlwasser tritt am Flansch aus", "fluid"),
    ("Druck fällt wegen einer Leckage ab", "fluid"),
    ("Schlauch weist einen Riss auf", "fluid"),
    ("Tank verliert langsam Flüssigkeit", "fluid"),
    ("Dichtung am Zylinder ist defekt", "fluid"),
    ("Ölfilm unter dem Getriebe sichtbar", "fluid"),
    ("Ventil lässt Medium unkontrolliert durch", "fluid"),
    ("Leitung schwitzt an der Verschraubung", "fluid"),
    ("Pumpe fördert wegen Luft im System schlecht", "fluid"),
    ("Hydraulikdruck sinkt nach dem Abschalten", "fluid"),
    ("Kühlmittelstand ist deutlich zu niedrig", "fluid"),
    ("Flanschdichtung muss ersetzt werden", "fluid"),
    ("Zylinder verliert Hydrauliköl", "fluid"),
    ("Temperatursensor liefert falsche Werte", "sensorik"),
    ("Drucksensor zeigt dauerhaft null", "sensorik"),
    ("Encoder meldet sprunghafte Positionen", "sensorik"),
    ("Näherungsschalter erkennt das Werkstück nicht", "sensorik"),
    ("Messwert des Sensors driftet langsam", "sensorik"),
    ("Lichtschranke bleibt ständig belegt", "sensorik"),
    ("Drehzahlsignal fällt zeitweise aus", "sensorik"),
    ("Füllstandsensor reagiert verzögert", "sensorik"),
    ("Kamera erkennt die Markierung nicht", "sensorik"),
    ("Positionsgeber liefert unplausible Werte", "sensorik"),
    ("Sensorstecker verursacht Messaussetzer", "sensorik"),
    ("Druckmessung springt ohne Prozessänderung", "sensorik"),
    ("Endschalter meldet eine falsche Position", "sensorik"),
    ("Temperaturwert bleibt konstant stehen", "sensorik"),
    ("Encoder verliert einzelne Impulse", "sensorik"),
    ("Messsystem ist nicht mehr kalibriert", "sensorik"),
]
WARTUNGSDATEN_TEXTE = [text for text, _ in WARTUNGSDATEN]
WARTUNGSDATEN_LABELS = [label for _, label in WARTUNGSDATEN]

#%% Daten und Hilfsfunktionen für Embeddings

import numpy as np
from sklearn.decomposition import TruncatedSVD

SAETZE = [
    "pumpe vibriert lager defekt",
    "pumpe rattert lager verschlissen",
    "motor vibriert lager heiß",
    "motor brummt lager beschädigt",
    "getriebe rattert zahnrad beschädigt",
    "getriebe vibriert welle unwucht",
    "ventil leckt dichtung defekt",
    "pumpe verliert öl dichtung undicht",
    "hydraulik verliert druck leitung undicht",
    "flansch verliert kühlwasser dichtung defekt",
    "sensor meldet temperatur falsch",
    "sensor liefert druck unplausibel",
    "encoder meldet position falsch",
    "lichtschranke erkennt werkstück nicht",
    "ersatzteile liegen im lager",
    "material kommt aus dem lager",
    "lagerbestand im lager ist niedrig",
    "techniker holt dichtung aus lager",
    "pumpe fördert wasser zuverlässig",
    "motor treibt pumpe zuverlässig an",
    "ventil regelt druck im system",
    "sensor misst druck im system",
    "encoder misst position der welle",
    "techniker prüft lager und welle",
]

def korpus_tokens():
    # Jeder Satz wird als eigene Tokenliste zurückgegeben.
    return [tokenisiere(satz) for satz in SAETZE]

def vokabular():
    # Das Vokabular wird deterministisch alphabetisch sortiert.
    woerter = sorted({w for satz in korpus_tokens() for w in satz})
    return woerter, {wort: i for i, wort in enumerate(woerter)}

def kontextpaare(fenster=2):
    # Zentrum-Kontext-Paare bilden die Grundlage vieler Embeddings.
    woerter, index = vokabular()
    paare = []
    for satz in korpus_tokens():
        for i, wort in enumerate(satz):
            links = max(0, i - fenster)
            rechts = min(len(satz), i + fenster + 1)
            for j in range(links, rechts):
                if i != j:
                    paare.append((index[wort], index[satz[j]]))
    return woerter, paare

def kookkurrenz(fenster=2):
    # Gerichtete Kookkurrenzen werden in einer Matrix gezählt.
    woerter, paare = kontextpaare(fenster)
    matrix = np.zeros((len(woerter), len(woerter)))
    for mitte, kontext in paare:
        matrix[mitte, kontext] += 1
    return woerter, matrix

def ppmi(matrix):
    # Positive PMI hebt überzufällig häufige Paare hervor.
    gesamt = matrix.sum()
    zeilen = matrix.sum(axis=1, keepdims=True)
    spalten = matrix.sum(axis=0, keepdims=True)
    erwartet = zeilen @ spalten / max(gesamt, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        werte = np.log2(matrix / erwartet)
    werte[~np.isfinite(werte)] = 0
    return np.maximum(werte, 0)

def svd_embedding(dimension=6):
    # PPMI wird mit einer abgeschnittenen SVD verdichtet.
    woerter, matrix = kookkurrenz()
    modell = TruncatedSVD(
        n_components=dimension,
        random_state=7,
    )
    vektoren = modell.fit_transform(ppmi(matrix))
    return woerter, vektoren

def kosinus(a, b):
    # Ein kleiner Schutz verhindert Division durch null.
    nenner = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / nenner) if nenner else 0.0

def nachbarn(wort, woerter, vektoren, anzahl=4):
    # Die ähnlichsten Wörter werden ohne das Suchwort ausgegeben.
    index = woerter.index(wort)
    werte = [kosinus(vektoren[index], v) for v in vektoren]
    rang = np.argsort(werte)[::-1]
    result = [(woerter[i], round(werte[i], 3))
              for i in rang if i != index]
    return result[:anzahl]

def wortvektor(wort, woerter, vektoren):
    # Unbekannte Wörter erhalten hier einen Nullvektor.
    if wort not in woerter:
        return np.zeros(vektoren.shape[1])
    return vektoren[woerter.index(wort)]

def satzvektor(text, woerter, vektoren, gewichte=None):
    # Mean Pooling bildet aus Wortvektoren einen Satzvektor.
    tokens = tokenisiere(text)
    teile = []
    faktoren = []
    for wort in tokens:
        if wort in woerter:
            teile.append(wortvektor(wort, woerter, vektoren))
            faktor = 1.0 if gewichte is None else gewichte.get(wort, 1.0)
            faktoren.append(faktor)
    if not teile:
        return np.zeros(vektoren.shape[1])
    return np.average(teile, axis=0, weights=faktoren)

#%% Daten und Hilfsfunktionen für PyTorch

DL_TEXTE = [
    "pumpe vibriert stark",
    "motorlager ist heiß",
    "getriebe rattert laut",
    "welle hat eine unwucht",
    "lager ist verschlissen",
    "motor brummt ungewöhnlich",
    "pumpe verliert öl",
    "ventil ist undicht",
    "dichtung am flansch leckt",
    "leitung verliert kühlwasser",
    "hydraulik verliert druck",
    "schlauch ist gerissen",
    "sensor meldet falsche werte",
    "encoder liefert keine position",
    "lichtschranke erkennt nichts",
    "temperaturfühler ist unplausibel",
    "drucksensor fällt zeitweise aus",
    "anzeige meldet einen fehler",
]
DL_LABELS = [
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2,
]
KLASSEN = ["mechanik", "leckage", "sensorik"]
PAD = "<pad>"
UNK = "<unk>"

def setze_seed(seed=7):
    # Ein Seed macht die kleinen Lehrbeispiele reproduzierbar.
    torch.manual_seed(seed)
    torch.set_num_threads(1)

def baue_vokabular():
    # Padding und unbekannte Wörter erhalten feste Indizes.
    woerter = sorted({w for text in DL_TEXTE for w in tokenisiere(text)})
    liste = [PAD, UNK] + woerter
    return liste, {wort: i for i, wort in enumerate(liste)}

def kodiere(text, index, laenge=7):
    # Sequenzen werden abgeschnitten oder rechts mit Padding ergänzt.
    ids = [index.get(w, index[UNK]) for w in tokenisiere(text)]
    ids = ids[:laenge] + [index[PAD]] * max(0, laenge - len(ids))
    return ids

def sequenzdaten(laenge=7):
    # Alle Texte werden als Tensoren mit identischer Länge ausgegeben.
    woerter, index = baue_vokabular()
    x = torch.tensor([kodiere(t, index, laenge) for t in DL_TEXTE])
    y = torch.tensor(DL_LABELS)
    return woerter, index, x, y

def bow_daten():
    # Ein Bag-of-Words-Tensor dient als Baseline für ein MLP.
    woerter, index = baue_vokabular()
    x = torch.zeros(len(DL_TEXTE), len(woerter))
    for zeile, text in enumerate(DL_TEXTE):
        for wort in tokenisiere(text):
            x[zeile, index.get(wort, index[UNK])] += 1
    return woerter, index, x, torch.tensor(DL_LABELS)

def trainiere(modell, x, y, epochen=180, rate=0.03):
    # Vollbatch-Training hält die Beispiele kurz und deterministisch.
    opt = torch.optim.Adam(modell.parameters(), lr=rate)
    verlauf = []
    for epoche in range(epochen + 1):
        opt.zero_grad()
        logits = modell(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if epoche in {0, epochen // 2, epochen}:
            verlauf.append((epoche, float(loss.detach())))
    return verlauf


#%% Hilfsfunktionen für Transformer

def positionskodierung(laenge, breite):
    # Sinus und Kosinus kodieren jede Position deterministisch.
    pos = torch.arange(laenge).unsqueeze(1)
    basis = -math.log(10000.0) / breite
    teiler = torch.exp(torch.arange(0, breite, 2) * basis)
    pe = torch.zeros(laenge, breite)
    pe[:, 0::2] = torch.sin(pos * teiler)
    pe[:, 1::2] = torch.cos(pos * teiler)
    return pe

class Klassifikator(torch.nn.Module):
    # Ein kleiner Encoder klassifiziert über das CLS-Token.
    def __init__(self, vokabular, klassen=3, breite=24, koepfe=4):
        super().__init__()
        self.emb = torch.nn.Embedding(vokabular, breite, padding_idx=0)
        self.register_buffer("pos", positionskodierung(12, breite))
        schicht = torch.nn.TransformerEncoderLayer(
            breite,
            koepfe,
            48,
            0.0,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(schicht, 2)
        self.kopf = torch.nn.Linear(breite, klassen)

    def forward(self, ids):
        # Eine Maske verhindert Aufmerksamkeit auf Paddingpositionen.
        maske = ids.eq(0)
        x = self.emb(ids) + self.pos[: ids.shape[1]]
        x = self.encoder(x, src_key_padding_mask=maske)
        return self.kopf(x[:, 0])
    
def genauigkeit(logits, y):
    # Accuracy genügt für den Funktionsnachweis des Lehrbeispiels.
    return float((logits.argmax(1) == y).float().mean())

#%% Daten für SFT-Beispiel in Kapitel 13

# 30 Wartungsmeldungen
wartungsmeldungen = [
    ("Pumpe 4 verliert Kühlwasser. Sofort prüfen.",
     "leckage", "hoch"),
    ("Hydraulikleitung tropft leicht.",
     "leckage", "niedrig"),
    ("Öl tritt stark aus der Pumpe aus.",
     "leckage", "hoch"),
    ("Unter Maschine 8 befindet sich eine kleine Ölpfütze.",
     "leckage", "mittel"),
    ("Kühlmittelschlauch ist feucht.",
     "leckage", "niedrig"),
    ("Dichtung an Ventil 3 verliert sichtbar Flüssigkeit.",
     "leckage", "mittel"),
    ("Großer Wasserverlust am Pumpengehäuse.",
     "leckage", "hoch"),
    ("Anschlussleitung zeigt geringe Undichtigkeit.",
     "leckage", "niedrig"),
    ("Hydrauliköl läuft kontinuierlich aus.",
     "leckage", "hoch"),
    ("Kleine Leckstelle am Kühlkreislauf entdeckt.",
     "leckage", "mittel"),

    ("Sensor 8 liefert falsche Messwerte.",
     "sensorik", "mittel"),
    ("Temperatursensor zeigt keine Werte mehr.",
     "sensorik", "hoch"),
    ("Drucksensor schwankt geringfügig.",
     "sensorik", "niedrig"),
    ("Füllstandssensor meldet dauerhaft 100 Prozent.",
     "sensorik", "mittel"),
    ("Positionssensor ist vollständig ausgefallen.",
     "sensorik", "hoch"),
    ("Messwert des Sensors weicht leicht vom Soll ab.",
     "sensorik", "niedrig"),
    ("Drehzahlsensor liefert stark springende Werte.",
     "sensorik", "mittel"),
    ("Temperaturmessung ist offensichtlich fehlerhaft.",
     "sensorik", "hoch"),
    ("Sensor reagiert gelegentlich verzögert.",
     "sensorik", "niedrig"),
    ("Druckmessung ist seit heute unplausibel.",
     "sensorik", "mittel"),

    ("Motorlager rattert laut. Austausch dringend.",
     "mechanik", "hoch"),
    ("Lager erzeugt leichte Laufgeräusche.",
     "mechanik", "niedrig"),
    ("Getriebe vibriert deutlich stärker als üblich.",
     "mechanik", "mittel"),
    ("Antriebswelle schlägt stark.",
     "mechanik", "hoch"),
    ("Leises Schleifgeräusch am Lüfterrad.",
     "mechanik", "niedrig"),
    ("Kupplung verursacht deutliche Vibrationen.",
     "mechanik", "mittel"),
    ("Motorlager ist blockiert.",
     "mechanik", "hoch"),
    ("Getriebe läuft etwas unruhig.",
     "mechanik", "niedrig"),
    ("Riemenantrieb erzeugt starke Geräusche.",
     "mechanik", "mittel"),
    ("Welle weist sichtbares mechanisches Spiel auf.",
     "mechanik", "hoch"),
]

# 9 unbekannte Wartungsmeldungen für den Vorher-Nachher-Test
wartungsmeldungen_testdaten = [
    (
        "Unter Pumpe 9 sammelt sich Hydrauliköl.",
        "leckage",
        "mittel",
    ),
    (
        "Am Kühlkreislauf tritt viel Wasser aus.",
        "leckage",
        "hoch",
    ),
    (
        "An einer Dichtung ist nur leichte Feuchtigkeit sichtbar.",
        "leckage",
        "niedrig",
    ),
    (
        "Der Temperatursensor zeigt dauerhaft 180 Grad.",
        "sensorik",
        "hoch",
    ),
    (
        "Ein Drucksensor schwankt leicht um den Sollwert.",
        "sensorik",
        "niedrig",
    ),
    (
        "Der Füllstandssensor liefert unplausible Werte.",
        "sensorik",
        "mittel",
    ),
    (
        "Das Motorlager verursacht starke Schlaggeräusche.",
        "mechanik",
        "hoch",
    ),
    (
        "Das Getriebe läuft etwas unruhig.",
        "mechanik",
        "niedrig",
    ),
    (
        "Die Antriebswelle vibriert deutlich stärker als üblich.",
        "mechanik",
        "mittel",
    ),
]

#%% Daten für generative Systeme, RAG und Dialoge

DOKUMENTE = [
    {
        "id": "HB-17",
        "titel": "Lager prüfen",
        "text": (
            "Bei starkem Lagergeräusch Anlage stoppen. "
            "Lager prüfen und Schmierung dokumentieren."
        ),
    },
    {
        "id": "HB-23",
        "titel": "Leckage behandeln",
        "text": (
            "Bei Kühlwasserverlust Leckstelle sichern. "
            "Dichtung prüfen und Flüssigkeit auffangen."
        ),
    },
    {
        "id": "HB-31",
        "titel": "Sensor prüfen",
        "text": (
            "Bei unplausiblen Messwerten Sensor reinigen. "
            "Anschließend Referenzmessung durchführen."
        ),
    },
    {
        "id": "HB-42",
        "titel": "Vibration bewerten",
        "text": (
            "Starke Vibrationen erfordern eine Sichtprüfung. "
            "Befestigung und Ausrichtung kontrollieren."
        ),
    },
]

# Synthetische Berichte für weitere Beispiele bereitstellen.
BERICHTE = [
    "pumpe vier verliert kühlwasser",
    "motorlager rattert sehr laut",
    "sensor liefert unplausible werte",
    "anlage zeigt starke vibrationen",
    "dichtung an pumpe zwei ist undicht",
    "lager von motor drei wird heiß",
]

FAQ = [
    (
        "Wann muss ein stark vibrierendes Lager geprüft werden?",
        (
            "Ein stark vibrierendes Lager soll innerhalb von zwei Wochen "
            "geprüft werden."
        ),
    ),
    (
        "Was ist bei unplausiblen Sensorwerten zu tun?",
        "Zuerst sind Sensorleitung und Steckverbindungen zu prüfen.",
    ),
    (
        "Wie wird bei einer undichten Dichtung vorgegangen?",
        "Die Anlage wird gesichert und die Dichtung kurzfristig ersetzt.",
    ),
]

DIALOGE = [
    ("Pumpe 4 vibriert stark", "stoerung"),
    ("Sensor 7 zeigt falsche Werte", "stoerung"),
    ("Das Ventil ist undicht", "stoerung"),
    ("Wann ist die Wartung geplant", "termin"),
    ("Nenne mir den nächsten Termin", "termin"),
    ("Wann findet die Inspektion statt", "termin"),
    ("Zeige den Status des Auftrags", "status"),
    ("Wie weit ist die Bearbeitung", "status"),
    ("Ist der Auftrag schon abgeschlossen", "status"),
    ("Danke für die Hilfe", "abschluss"),
    ("Bitte schließe den Dialog", "abschluss"),
    ("Auf Wiedersehen", "abschluss"),
]
#%% Daten für die Evaluation

Y_WAHR = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
Y_PROB = [0.92, 0.28, 0.81, 0.63, 0.45, 0.11, 0.74, 0.39,
          0.56, 0.22, 0.67, 0.52]

RELEVANT = {"D2", "D4", "D7"}
RANGFOLGE = ["D4", "D1", "D7", "D5", "D2"]
RELEVANZEN = [3, 0, 2, 1, 3]

REFERENZ = "das lager innerhalb von zwei wochen prüfen"
KANDIDAT = "lager binnen zwei wochen kontrollieren"

BEWERTER_A = [3, 2, 3, 1, 2, 3, 2, 1]
BEWERTER_B = [3, 2, 2, 1, 2, 3, 1, 1]

#%% Daten für verantwortungsvolle und sichere Systeme

MELDUNGEN = [
    {
        "id": "M-101",
        "werk": "Nord",
        "text": "Mia meldet: Pumpe 4 vibriert stark. "
                "E-Mail mia@example.org",
    },
    {
        "id": "M-102",
        "werk": "Süd",
        "text": "Omar meldet: Filterwechsel nächste Woche. "
                "Telefon +49 170 1234567",
    },
    {
        "id": "M-103",
        "werk": "Nord",
        "text": "Motor 7 wird heiß, Ursache noch unklar.",
    },
]

#%% Daten für produktive NLP-Systeme

TRAIN_X = [
    "Pumpe vibriert stark und muss sofort geprüft werden",
    "Motor überhitzt und die Anlage steht",
    "Ventil ist undicht, Gefahr für die Umgebung",
    "Lager macht laute Geräusche und wird heiß",
    "Filterwechsel ist für nächste Woche geplant",
    "Anlage läuft ruhig und ohne Auffälligkeiten",
    "Routinekontrolle wurde erfolgreich abgeschlossen",
    "Dichtung wird beim nächsten Service erneuert",
]
TRAIN_Y = [1, 1, 1, 1, 0, 0, 0, 0]

ANFRAGEN = [
    {"id": "M-201", "text": "Pumpe 4 vibriert stark"},
    {"id": "M-202", "text": "Filterwechsel nächste Woche"},
    {"id": "M-203", "text": "Motor wird ungewöhnlich heiß"},
]

PRODUKTIONS_DOKUMENTE = [
    "Pumpe 4: Lager prüfen und Schwingung messen",
    "Motor 7: Kühlung kontrollieren und Temperatur erfassen",
    "Filter 2: Wechsel im nächsten Wartungsfenster",
    "Ventil 3: Dichtung prüfen und Leckage dokumentieren",
]
