# Unicode und Schreibweise für Vergleiche vereinheitlichen.
import unicodedata

text_1 = "Straße"
text_2 = "STRASSE"
print(text_1.casefold() == text_2.casefold())

text_3 = "Störung"
text_4 = unicodedata.normalize("NFC", text_3)
print(text_3 == "Störung")
print(text_4 == "Störung")