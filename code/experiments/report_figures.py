"""Create the model-evaluation figures used in the final report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)

def main(output: Path) -> None:
    from joblib import load

    OUT = output / "figures"
    SCORES = output / "metrics" / "holdout_scores.parquet"
    METRICS = output / "metrics" / "model_metrics.csv"
    MODEL = output / "models" / "selected_fraud_pipeline.joblib"
    METADATA = output / "models" / "selected_model_metadata.json"
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(SCORES)
    metrics = pd.read_csv(METRICS)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    config = metadata["selected_feature_config"]
    algorithm = metadata["selected_algorithm"]
    threshold = float(metadata["threshold"])
    score_column = f"{config}__{algorithm}"
    y = table["is_fraud"].to_numpy(dtype="int8")
    score = table[score_column].to_numpy(dtype=float)
    predicted = (score >= threshold).astype("int8")
    ap = float(
        metrics.loc[
            metrics.feature_config.eq(config) & metrics.algorithm.eq(algorithm),
            "holdout_ap",
        ].iloc[0]
    )

    # A compact report plot compares the selected model with the raw-feature
    # XGBoost baseline, avoiding low-performing configurations that obscure it.
    fig, ax = plt.subplots(figsize=(145 / 25.4, 82 / 25.4))
    for column, label, color in (
        (score_column, "Raw + extra, logistic regression", "#B2182B"),
        ("raw__xgboost", "Raw features, XGBoost", "#2166AC"),
    ):
        candidate_scores = table[column].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y, candidate_scores)
        candidate_ap = float(
            metrics.loc[
                metrics.feature_config.eq(column.split("__")[0])
                & metrics.algorithm.eq(column.split("__")[1]),
                "holdout_ap",
            ].iloc[0]
        )
        ax.plot(recall, precision, lw=1.5, color=color,
                label=f"{label} (AP={candidate_ap:.3f})")
    prevalence = float(y.mean())
    ax.axhline(prevalence, color="#666666", lw=0.8, ls="--",
               label=f"Holdout prevalence ({prevalence:.5f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.set_title("Precision--recall on the internal holdout", fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.5)
    ax.grid(color="#E0E0E0", linewidth=0.3)
    fig.savefig(OUT / "selected_model_pr_comparison.pdf")
    plt.close(fig)

    # Confusion matrix with raw counts; rows are true labels and columns are alerts.
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(95 / 25.4, 76 / 25.4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual label")
    ax.set_title(f"Holdout confusion matrix, threshold={threshold:.6f}", fontweight="bold")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center", color="#222222")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Transactions")
    fig.savefig(OUT / "selected_model_confusion_matrix.pdf")
    plt.close(fig)

    # ROC curve uses continuous scores from the same untouched holdout partition.
    fpr, tpr, _ = roc_curve(y, score)
    roc_auc = float(
        metrics.loc[
            metrics.feature_config.eq(config) & metrics.algorithm.eq(algorithm),
            "holdout_roc_auc",
        ].iloc[0]
    )
    fig, ax = plt.subplots(figsize=(95 / 25.4, 76 / 25.4))
    ax.plot(fpr, tpr, color="#2166AC", lw=1.3, label=f"ROC-AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls="--", label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate", ylabel="True-positive rate")
    ax.set_title("Selected model on the internal holdout", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(color="#E0E0E0", linewidth=0.3)
    fig.savefig(OUT / "selected_model_roc_curve.pdf")
    plt.close(fig)

    # Standardized logistic coefficients are descriptive model weights, not causal effects.
    pipeline = load(MODEL)
    preprocess = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        raise TypeError("Selected model has no linear coefficients")
    names = preprocess.get_feature_names_out()
    coefficients = estimator.coef_[0]
    top = np.argsort(np.abs(coefficients))[-15:]
    order = top[np.argsort(np.abs(coefficients[top]))]
    labels = [names[i].replace("numeric__", "").replace("categorical__", "") for i in order]
    values = coefficients[order]
    colors = ["#B2182B" if value > 0 else "#2166AC" for value in values]
    fig, ax = plt.subplots(figsize=(175 / 25.4, 105 / 25.4))
    ax.barh(np.arange(len(order)), values, color=colors)
    ax.set_yticks(np.arange(len(order)), labels)
    ax.axvline(0, color="#222222", lw=0.6)
    ax.set_xlabel("Logistic coefficient (standardized numeric inputs)")
    ax.set_title("Largest absolute coefficients in the final model", fontweight="bold")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.3)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.005, "Weights describe this fitted model and do not imply causation.", ha="center", fontsize=7)
    fig.savefig(OUT / "selected_model_coefficients.pdf")
    plt.close(fig)

    print(f"Saved model figures; AP={ap:.6f}, ROC-AUC={roc_auc:.6f}, threshold={threshold:.9f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output"))
    main(parser.parse_args().output)
