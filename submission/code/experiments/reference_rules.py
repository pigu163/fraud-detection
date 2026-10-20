"""Evaluate simple reference rules on the saved internal holdout.

The holdout row positions and labels are read from the training-only artifact
created by train_ablation.py.  The external test file is never opened.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    predicted = (scores >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan,
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "alert_count": int(predicted.sum()),
        "alert_rate": float(predicted.mean()),
        "false_alerts_per_100k": float(100_000 * fp / len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("train_fraud.parquet"))
    parser.add_argument("--holdout", type=Path, default=Path("output/metrics/holdout_scores.parquet"))
    parser.add_argument("--output", type=Path, default=Path("output/metrics/reference_rules.csv"))
    args = parser.parse_args()

    scores = pd.read_parquet(args.holdout)
    positions = scores["row_position"].to_numpy(dtype=np.int64)
    y_true = scores["is_fraud"].to_numpy(dtype=np.int8)
    train = pd.read_parquet(
        args.train,
        columns=["transaction_amount", "initiater_balance_before", "is_flagged_fraud"],
    )
    if positions.max(initial=-1) >= len(train):
        raise ValueError("holdout row positions exceed the training table")
    subset = train.iloc[positions]
    amount = subset["transaction_amount"].to_numpy(dtype=float)
    balance = subset["initiater_balance_before"].to_numpy(dtype=float)
    full_balance = ((balance > 0) & (np.abs(amount - balance) <= 0.01)).astype(float)
    supplied_flag = subset["is_flagged_fraud"].to_numpy(dtype=float)
    all_negative = np.zeros_like(y_true, dtype=float)

    rows = []
    for name, rule_scores in (
        ("all_negative", all_negative),
        ("supplied_is_flagged_fraud", supplied_flag),
        ("full_balance_amount_match", full_balance),
    ):
        row = {"reference": name, **evaluate(y_true, rule_scores)}
        rows.append(row)
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
