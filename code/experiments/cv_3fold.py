"""Leakage-aware 3-fold check for the raw and engineered feature groups.

Each outer fold is evaluated once.  The threshold is selected from an inner
calibration split inside the outer training rows, so the validation fold stays
unseen until scoring.
"""

from __future__ import annotations

import gc
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from train_ablation import (
    ALGORITHMS,
    EXTRA_COLUMNS,
    RANDOM_STATE,
    RAW_COLUMNS,
    TRAIN_DEFAULT,
    add_extra_features,
    best_f1_threshold,
    evaluate_scores,
    make_pipeline,
    save_pdf,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
CONFIGS = {
    "raw": RAW_COLUMNS,
    "raw_plus_extra": RAW_COLUMNS + EXTRA_COLUMNS,
}


def run_cv(train_path: Path = TRAIN_DEFAULT) -> pd.DataFrame:
    train = pd.read_parquet(train_path).reset_index(drop=True)
    y = train["is_fraud"].to_numpy(dtype="int8")
    enriched = add_extra_features(train)
    frames = {
        "raw": enriched.loc[:, RAW_COLUMNS].copy(),
        "raw_plus_extra": enriched.loc[:, RAW_COLUMNS + EXTRA_COLUMNS].copy(),
    }

    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, object]] = []

    for fold, (outer_train_idx, validation_idx) in enumerate(
        outer.split(np.zeros(len(y)), y), start=1
    ):
        # The inner split is only for threshold selection.
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=0.20, random_state=RANDOM_STATE + fold
        )
        fit_rel, calibration_rel = next(
            inner.split(np.zeros(len(outer_train_idx)), y[outer_train_idx])
        )
        fit_idx = outer_train_idx[fit_rel]
        calibration_idx = outer_train_idx[calibration_rel]

        for config, columns in CONFIGS.items():
            X = frames[config]
            for algorithm in ALGORITHMS:
                # ponytail: use the same small baseline settings as the main
                # ablation; this run checks split stability, not tuning.
                fit_positive_weight = (
                    len(fit_idx) - int(y[fit_idx].sum())
                ) / max(int(y[fit_idx].sum()), 1)
                threshold_model = make_pipeline(columns, algorithm, fit_positive_weight)
                threshold_model.fit(X.iloc[fit_idx], y[fit_idx])
                calibration_scores = threshold_model.predict_proba(
                    X.iloc[calibration_idx]
                )[:, 1]
                threshold = best_f1_threshold(
                    y[calibration_idx], calibration_scores
                )
                del threshold_model, calibration_scores
                gc.collect()

                # Refit on all outer-training rows before scoring the unseen fold.
                outer_positive_weight = (
                    len(outer_train_idx) - int(y[outer_train_idx].sum())
                ) / max(int(y[outer_train_idx].sum()), 1)
                model = make_pipeline(columns, algorithm, outer_positive_weight)
                model.fit(X.iloc[outer_train_idx], y[outer_train_idx])
                validation_scores = model.predict_proba(X.iloc[validation_idx])[:, 1]
                result = evaluate_scores(
                    y[validation_idx], validation_scores, threshold
                )
                rows.append(
                    {
                        "fold": fold,
                        "feature_config": config,
                        "algorithm": algorithm,
                        "threshold": threshold,
                        "validation_rows": len(validation_idx),
                        "validation_positive": int(y[validation_idx].sum()),
                        "average_precision": result["average_precision"],
                        "roc_auc": result["roc_auc"],
                        "precision": result["precision"],
                        "recall": result["recall"],
                        "f1": result["f1"],
                        "tn": result["tn"],
                        "fp": result["fp"],
                        "fn": result["fn"],
                        "tp": result["tp"],
                        "alert_count": result["alert_count"],
                    }
                )
                del model, validation_scores
                gc.collect()
                print(
                    f"fold={fold} config={config} algorithm={algorithm} "
                    f"AP={result['average_precision']:.4f} F1={result['f1']:.4f} "
                    f"threshold={threshold:.6g}",
                    flush=True,
                )

    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    configs = list(CONFIGS)
    algorithms = list(ALGORITHMS)
    x = np.arange(len(configs))
    width = 0.24
    colors = ["#2166AC", "#B2182B", "#1B7837"]
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 78 / 25.4))
    for ax, metric, title in zip(
        axes,
        ("average_precision", "f1"),
        ("3-fold Average Precision", "3-fold fraud-class F1"),
    ):
        for i, (algorithm, color) in enumerate(zip(algorithms, colors)):
            selected = summary[summary.algorithm.eq(algorithm)].set_index(
                "feature_config"
            ).loc[configs]
            ax.bar(
                x + (i - 1) * width,
                selected[f"{metric}_mean"],
                width,
                yerr=selected[f"{metric}_std"],
                capsize=2,
                color=color,
                label=algorithm.replace("_", " ").title(),
            )
        ax.set_xticks(x, [c.replace("_", " + ") for c in configs], rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.3)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.text(
        0.5,
        0.005,
        "Bars show mean ± SD across three unseen validation folds; thresholds were chosen inside each outer training fold.",
        ha="center",
        fontsize=7.5,
    )
    save_pdf(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    metrics_dir = args.output / "metrics"
    figures_dir = args.output / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = run_cv(args.train)
    summary = (
        fold_metrics.groupby(["feature_config", "algorithm"], as_index=False)
        .agg(
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", "std"),
        )
    )
    fold_metrics.to_csv(metrics_dir / "cv3fold_fold_metrics.csv", index=False)
    summary.to_csv(metrics_dir / "cv3fold_summary.csv", index=False)
    fold_metrics.to_parquet(metrics_dir / "cv3fold_fold_metrics.parquet", index=False)
    summary.to_parquet(metrics_dir / "cv3fold_summary.parquet", index=False)
    plot_summary(summary, figures_dir / "cv3fold_summary.pdf")
    print("Saved 3-fold metrics to", metrics_dir, flush=True)
    print("Saved 3-fold figure to", figures_dir / "cv3fold_summary.pdf", flush=True)


if __name__ == "__main__":
    main()
