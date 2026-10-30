# URLs, Adressen, Hashtags, Erwähnungen und Emojis markieren
import spacy
# Leere deutsche Pipeline mit Tokenizer erzeugen.
nlp = spacy.blank("de")
text = ("Mail an service@firma.de, siehe https://firma.de #Wartung @Team 😊")
print("Text original: ", text)
doc = nlp(text)
# Nur explizit bekannte Emojis kategorisieren.
emojis = {"😊", "😞", "😐"}
treffer = []
i = 0
while i < len(doc):
    token = doc[i]
    if token.like_email:
        label = "EMAIL"
        span = doc[i:i + 1]
    elif token.like_url:
        label = "URL"
        span = doc[i:i + 1]
    elif token.text == "#" and i + 1 < len(doc):
        label = "HASHTAG"
        span = doc[i:i + 2]
        i += 1
    elif token.text.startswith("@"):
        label = "MENTION"
        span = doc[i:i + 1]
    elif token.text in emojis:
        label = "EMOJI"
        span = doc[i:i + 1]
    else:
        i += 1
        continue
    treffer.append((label, span.text, span.start_char, span.end_char))
    i += 1
# Treffer im Originaltext durch Platzhalter ersetzen.
markiert = text
for label, _, start, ende in reversed(treffer):
    markiert = markiert[:start] + f"<{label}>" + markiert[ende:]
print("Text markiert: ", markiert)
print("Gefundene Sonderformen:")
for label, wert, _, _ in treffer:
    anzeige = wert
    if label == "EMOJI":
        anzeige = f"U+{ord(wert):04X}"
    print(f"   {label:<8}: {anzeige}")

