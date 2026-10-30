# Ein kurzer Wartungstext als Python-String
text = "Pumpe 4 zeigt starke Vibrationen."
tokens = text.rstrip(".").split()
anlage = " ".join(tokens[:2])

print(f"Text: {text}")
print(f"\nTokens: {tokens}")
print(f"Anlage: {anlage}")
print(f"Anzahl Tokens: {len(tokens)}")





