"""Custom XGBoost training script (SageMaker script mode).

SageMaker invokes this as:
    python train.py --num_round 100 --max_depth 5 --eta 0.2 ...

Channels are mounted at /opt/ml/input/data/<channel>; the model goes to /opt/ml/model.
"""
import argparse
import glob
import os

import numpy as np
import xgboost as xgb


def load_csv_dir(directory: str) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {directory}: {os.listdir(directory)}")
    arrays = [np.loadtxt(f, delimiter=",") for f in files]
    data = np.vstack(arrays)
    return data[:, 1:], data[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    # XGBoost hyperparameters
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--eval_metric", type=str, default="error")
    # SageMaker channels (env vars are injected by the framework container)
    parser.add_argument("--train", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--validation", type=str,
                        default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args, _ = parser.parse_known_args()

    print(f"args = {vars(args)}")

    X_train, y_train = load_csv_dir(args.train)
    X_val, y_val = load_csv_dir(args.validation)
    print(f"train shape: X={X_train.shape} y={y_train.shape}")
    print(f"val shape:   X={X_val.shape} y={y_val.shape}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": args.objective,
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "eval_metric": args.eval_metric,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        verbose_eval=10,
    )

    os.makedirs(args.model_dir, exist_ok=True)
    out_path = os.path.join(args.model_dir, "xgboost-model")
    booster.save_model(out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
