# Zeichen-N-Gramme als Bausteine von FastText untersuchen

def ngramme(wort, minimum=3, maximum=5):
    # Randzeichen markieren Wortanfang und Wortende.
    erweitert = f"<{wort}>"
    return {
        erweitert[i:i + n]
        for n in range(minimum, maximum + 1)
        for i in range(len(erweitert) - n + 1)
    }

wort_a = "lager"
wort_b = "lagerschaden"
a = ngramme(wort_a)
b = ngramme(wort_b)
print(f"{wort_a}: {len(a)} N-Gramme")
print(f"{wort_b}: {len(b)} N-Gramme")
print(f"Gemeinsam: {len(a & b)}")
print(", ".join(sorted(a & b)))
