# T5 formuliert unterschiedliche Aufgaben als Text-zu-Text-Problem
import re

text = "Pumpe 4 verliert stark Öl und muss morgen geprüft werden."
spanne = "verliert stark Öl"
eingabe = text.replace(spanne, "<extra_id_0>")
ziel = f"<extra_id_0> {spanne} <extra_id_1>"
klassifikation = "klassifiziere: " + text
zusammenfassung = "fasse zusammen: " + text
kurz = re.sub(r" und .*", "", text)
print("Span-Corruption:")
print(" Eingabe:", eingabe)
print(" Ziel:   ", ziel)
print("Text-zu-Text-Aufgaben:")
print(" ", klassifikation)
print(" ", zusammenfassung)
print(" Beispielziel:", kurz) 
