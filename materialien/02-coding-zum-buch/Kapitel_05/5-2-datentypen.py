# Werte aus einer CSV-Datei liegen zunächst als Strings vor
token_anzahl_text = "128"
konfidenz_text = "0.92"
dokument_id = str(4711)
print(type(token_anzahl_text))
print(isinstance(token_anzahl_text, int))
# Gültige Werte explizit konvertieren
try:
    token_anzahl = int(token_anzahl_text)
    konfidenz = float(konfidenz_text)
    print(type(token_anzahl))
    print(isinstance(token_anzahl, int))
    print(
        f"Dokument {dokument_id}: {token_anzahl} Tokens, "
        f"Konfidenz: {konfidenz:.0%}"
    )
except ValueError:
    print("Konvertierung fehlgeschlagen.")

# Fehlerhafte Eingabe absichern
try:
    fehlerwert = int("unbekannt")
except ValueError:
    fehlerwert = None
print(f"Ungültige Angabe: {fehlerwert}")