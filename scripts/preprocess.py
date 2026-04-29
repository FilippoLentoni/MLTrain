"""Generate synthetic classification data, split into train/validation/test, write CSV."""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

OUT_TRAIN = "/opt/ml/processing/output/train"
OUT_VAL = "/opt/ml/processing/output/validation"
OUT_TEST = "/opt/ml/processing/output/test"

for d in (OUT_TRAIN, OUT_VAL, OUT_TEST):
    os.makedirs(d, exist_ok=True)

X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    n_classes=2,
    random_state=42,
)
df = pd.DataFrame(X)
df.insert(0, "label", y)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

train_df.to_csv(f"{OUT_TRAIN}/train.csv", index=False, header=False)
val_df.to_csv(f"{OUT_VAL}/validation.csv", index=False, header=False)
test_df.to_csv(f"{OUT_TEST}/test.csv", index=False, header=False)

print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
