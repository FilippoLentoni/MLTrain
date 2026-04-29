"""Load trained XGBoost model + held-out test set; compute accuracy; write metrics.json."""
import os
import json
import glob
import tarfile
import numpy as np
import xgboost as xgb

MODEL_DIR = "/opt/ml/processing/input/model"
TEST_DIR = "/opt/ml/processing/input/test"
OUT_DIR = "/opt/ml/processing/output/metrics"

os.makedirs(OUT_DIR, exist_ok=True)

archive = os.path.join(MODEL_DIR, "model.tar.gz")
with tarfile.open(archive) as tar:
    tar.extractall(MODEL_DIR)

candidates = (
    glob.glob(f"{MODEL_DIR}/xgboost-model")
    + glob.glob(f"{MODEL_DIR}/*.bin")
    + glob.glob(f"{MODEL_DIR}/*.json")
    + glob.glob(f"{MODEL_DIR}/*.model")
)
if not candidates:
    raise FileNotFoundError(f"No model file found in {MODEL_DIR}: {os.listdir(MODEL_DIR)}")

booster = xgb.Booster()
booster.load_model(candidates[0])

test_csv = glob.glob(f"{TEST_DIR}/*.csv")[0]
data = np.loadtxt(test_csv, delimiter=",")
y_true = data[:, 0].astype(int)
X = data[:, 1:]

preds = booster.predict(xgb.DMatrix(X))
y_pred = (preds > 0.5).astype(int)

accuracy = float(np.mean(y_pred == y_true))
metrics = {
    "binary_classification_metrics": {
        "accuracy": {"value": accuracy, "standard_deviation": "NaN"}
    },
    "n_samples": int(len(y_true)),
}

with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"accuracy={accuracy:.4f} n={len(y_true)}")
