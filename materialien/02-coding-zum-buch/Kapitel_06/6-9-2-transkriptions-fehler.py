# ASR-Hypothesen mit Fachkontext neu bewerten.
hypothesen = [
    ("Pumpe für prüfen", 0.48),
    ("Pumpe vier prüfen", 0.42),
    ("Pumpe wir prüfen", 0.10),
]

# Fachliche Plausibilität als Gewicht ergänzen.
fachgewicht = {
    "Pumpe für prüfen": 0.20,
    "Pumpe vier prüfen": 0.90,
    "Pumpe wir prüfen": 0.10,
}

scores = []
for text, p_asr in hypothesen:
    score = p_asr * fachgewicht[text]
    scores.append((text, p_asr, score))

gesamt = sum(score for _, _, score in scores)
bewertet = [
    (text, p_asr, score / gesamt)
    for text, p_asr, score in scores
]

for text, p_asr, p_neu in bewertet:
    print(
        f"{text:<20} "
        f"ASR={p_asr:.2f} "
        f"neu={p_neu:.3f}"
    )

bester = max(bewertet, key=lambda eintrag: eintrag[2])
print("\nAusgewählt:", bester[0])