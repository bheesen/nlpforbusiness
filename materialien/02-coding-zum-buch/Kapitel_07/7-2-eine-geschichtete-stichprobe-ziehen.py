# Eine geschichtete Stichprobe nach Dokumentklasse ziehen
import pandas as pd

korpus = pd.DataFrame({
    "id": [f"B{i:02d}" for i in range(1, 13)],
    "klasse": ["Pumpe"] * 6 + ["Motor"] * 4 + ["Ventil"] * 2,
})

stichprobe = (
    korpus.groupby("klasse", group_keys=False)
    .sample(n=2, random_state=7)
    .sort_values(["klasse", "id"])
)
print("Gesamtkorpus:")
print(korpus["klasse"].value_counts().sort_index())
print("\nStichprobe:")
print(stichprobe.to_string(index=False))
