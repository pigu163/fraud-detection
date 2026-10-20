"""Post-hoc chronological sensitivity check using training data only.

The hour boundaries follow the observed labeled-data coverage and were not
pre-registered. Results are supplementary, not a blind prospective estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from train_ablation import (
    CONFIGS,
    add_extra_features,
    best_f1_threshold,
    evaluate_scores,
    make_pipeline,
)


def main(train_path: Path, output_dir: Path) -> None:
    data = pd.read_parquet(train_path)
    y = data["is_fraud"].to_numpy(dtype="int8")
    hour = data["hours_elapsed"].to_numpy()
    fit = hour <= 659
    calibration = (hour >= 660) & (hour <= 680)
    holdout = (hour >= 681) & (hour <= 718)
    late_fraud_only = hour >= 719

    X = add_extra_features(data).loc[:, CONFIGS["raw_plus_extra"]]
    positive_weight = float((fit.sum() - y[fit].sum()) / y[fit].sum())
    feature_sets = {
        "raw": CONFIGS["raw"],
        "raw_plus_extra": CONFIGS["raw_plus_extra"],
        "raw_plus_extra_no_hour_or_flag": [
            col for col in CONFIGS["raw_plus_extra"]
            if col not in ("hours_elapsed", "is_flagged_fraud")
        ],
    }
    variants = {}
    for name, columns in feature_sets.items():
        pipeline = make_pipeline(columns, "logistic_regression", positive_weight)
        pipeline.fit(X.loc[fit, columns], y[fit])
        calibration_scores = pipeline.predict_proba(X.loc[calibration, columns])[:, 1]
        threshold = best_f1_threshold(y[calibration], calibration_scores)
        holdout_scores = pipeline.predict_proba(X.loc[holdout, columns])[:, 1]
        late_scores = pipeline.predict_proba(X.loc[late_fraud_only, columns])[:, 1]
        variants[name] = {
            "features": columns,
            "threshold": threshold,
            "calibration_ap": float(average_precision_score(y[calibration], calibration_scores)),
            "holdout_metrics": evaluate_scores(y[holdout], holdout_scores, threshold),
            "late_fraud_only_detected": int((late_scores >= threshold).sum()),
        }

    results = {
        "analysis": "post_hoc_chronological_sensitivity",
        "algorithm": "logistic_regression",
        "fit_hours": "0-659",
        "calibration_hours": "660-680",
        "holdout_hours": "681-718",
        "fit_rows": int(fit.sum()),
        "fit_fraud": int(y[fit].sum()),
        "calibration_rows": int(calibration.sum()),
        "calibration_fraud": int(y[calibration].sum()),
        "holdout_rows": int(holdout.sum()),
        "holdout_fraud": int(y[holdout].sum()),
        "late_fraud_only_rows": int(late_fraud_only.sum()),
        "variants": variants,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "temporal_validation.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("train_fraud.parquet"))
    parser.add_argument("--output", type=Path, default=Path("output/metrics"))
    args = parser.parse_args()
    main(args.train, args.output)
