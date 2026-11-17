# Verwandte Dokumente gemeinsam einem Datensplit zuordnen
import hashlib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

korpus = pd.DataFrame({
    "id": [f"B{i}" for i in range(1, 9)],
    "anlage": ["P1", "P1", "P2", "P2", "M1", "M1", "V1", "V1"],
    "text": ["laut", "laut", "warm", "heiß", "klemmt", "stoppt",
             "undicht", "undicht"],
})
splitter = GroupShuffleSplit(test_size=0.25, random_state=4)
train_idx, test_idx = next(splitter.split(korpus, groups=korpus["anlage"]))
korpus["split"] = "train"
korpus.loc[test_idx, "split"] = "test"
korpus["hash"] = korpus["text"].map(
    lambda text: hashlib.sha256(text.encode()).hexdigest()[:6]
)
print(korpus.to_string(index=False))
print("\nDubletten in mehreren Splits:")
print(korpus.groupby("hash")["split"].nunique().gt(1).sum())
print("\nDubletten in mehreren Splits:")
print(korpus.groupby("hash")["split"].nunique().gt(1).sum())
