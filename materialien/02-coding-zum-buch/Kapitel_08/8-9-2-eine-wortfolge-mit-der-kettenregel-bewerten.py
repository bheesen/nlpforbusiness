# Eine Satzwahrscheinlichkeit mit der Kettenregel berechnen
from collections import Counter

# Derselbe Korpus wie in der vorangehenden Übung.
saetze = [
    ["die", "pumpe", "vibriert", "stark"],
    ["die", "pumpe", "verliert", "öl"],
    ["der", "motor", "wird", "heiß"],
    ["die", "pumpe", "wird", "geprüft"],
]
# Alle Wörter der vier Sätze zu einer Liste zusammenfassen.
tokens = [
    wort
    for satz in saetze
    for wort in satz
]
# Häufigkeiten aller einzelnen Wörter zählen.
unigramme = Counter(tokens)
# Wortpaare, also Bigramme, aus allen Sätzen erzeugen.
bigramme = Counter(
    (satz[i], satz[i + 1])
    for satz in saetze
    for i in range(len(satz) - 1)
)
# Wortdreier, also Trigramme, aus allen Sätzen erzeugen.
trigramme = Counter(
    (satz[i], satz[i + 1], satz[i + 2])
    for satz in saetze
    for i in range(len(satz) - 2)
)
# Wortvierer aus allen Sätzen erzeugen.
viergramme = Counter(
    tuple(satz[i:i + 4])
    for satz in saetze
    for i in range(len(satz) - 3)
)
# Wahrscheinlichkeit für das erste Wort aus seiner Häufigkeit bestimmen.
p_die = unigramme["die"] / len(tokens)
# Wahrscheinlichkeit von "pumpe" nach "die" bestimmen.
p_pumpe = (
    bigramme["die", "pumpe"]
    / unigramme["die"]
)
# Wahrscheinlichkeit von "vibriert" nach "die pumpe" bestimmen.
p_vibriert = (
    trigramme["die", "pumpe", "vibriert"]
    / bigramme["die", "pumpe"]
)
# Wahrscheinlichkeit von "stark" nach "die pumpe vibriert".
p_stark = (
    viergramme["die", "pumpe", "vibriert", "stark"]
    / trigramme["die", "pumpe", "vibriert"]
)
# Die vier Faktoren der Kettenregel zusammenstellen.
faktoren = [
    ("P(die)", p_die),
    ("P(pumpe | die)", p_pumpe),
    ("P(vibriert | die pumpe)", p_vibriert),
    ("P(stark | die pumpe vibriert)", p_stark),
]
# Mit 1 starten, da alle Faktoren schrittweise multipliziert werden.
wahrscheinlichkeit = 1.0

# Überschrift für eine übersichtliche Ausgabe erzeugen.
print("Faktor                               Wert")
print("-" * 43)
# Jeden Faktor ausgeben und zur Satzwahrscheinlichkeit multiplizieren.
for name, wert in faktoren:
    wahrscheinlichkeit *= wert
    print(f"{name:<35} {wert:>6.4f}")
# Ergebnis der vollständigen Kettenregel ausgeben.
print("-" * 43)
print(
    "P(die pumpe vibriert stark) = "
    f"{wahrscheinlichkeit:.4f}"
)
