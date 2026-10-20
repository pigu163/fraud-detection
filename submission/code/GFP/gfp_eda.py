"""Descriptive, training-only plots for prior-hour graph features."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graph_features import FEATURE_COLUMNS


BLUE = "#2166AC"
RED = "#B2182B"
BLACK = "#222222"
BUCKETS = ("0", "1", "2", "3–5", "6–10", "11+")
GRAPH_TYPES = ("CASH_OUT", "TRANSFER")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def _bucket_counts(values: pd.Series) -> pd.Categorical:
    if values.isna().any() or (values < 0).any():
        raise ValueError("Graph counts must be non-negative and non-missing")
    return pd.cut(
        values,
        bins=[-0.5, 0.5, 1.5, 2.5, 5.5, 10.5, np.inf],
        labels=BUCKETS,
    )


def summarize_gfp_by_label(
    train_df: pd.DataFrame, graph_features: pd.DataFrame
) -> pd.DataFrame:
    """Count all records in each feature bin; no modeling or statistical claim."""
    if len(train_df) != len(graph_features):
        raise ValueError("Training rows and graph feature rows differ")
    if not train_df.index.equals(graph_features.index):
        raise ValueError("Training rows and graph feature index differ")
    if not train_df["is_fraud"].isin([0, 1]).all():
        raise ValueError("is_fraud must be binary")

    results = []
    for feature in FEATURE_COLUMNS:
        bucket = _bucket_counts(graph_features[feature])
        work = pd.DataFrame(
            {
                "transaction_type": train_df["transaction_type"].to_numpy(),
                "bucket": bucket,
                "is_fraud": train_df["is_fraud"].to_numpy(),
            }
        )
        grouped = (
            work[work["transaction_type"].isin(GRAPH_TYPES)]
            .groupby(["transaction_type", "bucket"], observed=False)["is_fraud"]
            .agg(transaction_count="size", fraud_count="sum")
            .reset_index()
        )
        grouped["fraud_rate_pct"] = np.where(
            grouped["transaction_count"] > 0,
            100 * grouped["fraud_count"] / grouped["transaction_count"],
            np.nan,
        )
        grouped.insert(0, "feature", feature)
        results.append(grouped)
    return pd.concat(results, ignore_index=True)


def plot_gfp_feature(
    summary: pd.DataFrame,
    feature: str,
    output_dir: str | Path = "image/graph_topology_features",
    context: str = "All training hours",
    filename: str | None = None,
) -> Path:
    """Show label-conditional distributions and bin fraud rates by type."""
    if feature not in FEATURE_COLUMNS:
        raise ValueError(f"Unknown graph feature: {feature}")
    selected = summary[summary["feature"].eq(feature)]
    if selected.empty:
        raise ValueError("No summary rows for requested feature")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex="col")
    x = np.arange(len(BUCKETS))
    for col, transaction_type in enumerate(GRAPH_TYPES):
        table = (
            selected[selected["transaction_type"].eq(transaction_type)]
            .set_index("bucket")
            .reindex(BUCKETS)
        )
        total = table["transaction_count"].fillna(0).to_numpy()
        fraud = table["fraud_count"].fillna(0).to_numpy()
        legitimate = total - fraud
        for shift, counts, color, label in [
            (-0.18, legitimate, BLUE, "Legitimate"),
            (0.18, fraud, RED, "Fraud"),
        ]:
            share = counts / counts.sum() * 100 if counts.sum() else np.zeros_like(counts)
            axes[0, col].bar(x + shift, share, width=0.34, color=color, label=label)
        axes[0, col].set_title(transaction_type, weight="bold")
        axes[0, col].set_ylabel("Within-label share (%)" if col == 0 else "")
        axes[0, col].set_ylim(bottom=0)

        rate = table["fraud_rate_pct"].fillna(0).to_numpy()
        axes[1, col].bar(x, rate, color=RED, width=0.62)
        baseline = 100 * fraud.sum() / total.sum()
        axes[1, col].axhline(baseline, color=BLACK, linestyle="--", linewidth=0.9)
        axes[1, col].set_ylabel("Fraud rate (%)" if col == 0 else "")
        axes[1, col].set_xlabel("Distinct counterparties in earlier hours" if "fan" in feature else "Transactions in earlier hours")
        axes[1, col].set_xticks(x, BUCKETS)
        axes[1, col].set_ylim(bottom=0, top=max(rate.max(), baseline) * 1.25 + 0.001)
        # Exact denominators prevent a dramatic rate from hiding a tiny bin.
        for j, (positives, denominator, value) in enumerate(zip(fraud, total, rate)):
            if denominator:
                axes[1, col].annotate(
                    f"{int(positives)}/{int(denominator):,}",
                    (j, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=35,
                )
    axes[0, 0].legend(frameon=False, ncol=2)
    title = feature.replace("gfp_", "").replace("_", " ").capitalize()
    fig.suptitle(f"{title}: distribution and fraud rate", fontsize=12, weight="bold")
    fig.text(
        0.5,
        0.015,
        f"Training-set EDA only ({context}). Dashed lines: type fraud rate; labels: fraud / all.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=1.6, w_pad=1.2)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / (filename or feature)
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path
