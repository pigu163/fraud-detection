# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) categorical label distribution → assets/figures/BarCategorical/ → param inherit
# (b) temporal trend → assets/figures/LineTrend/ → param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import FuncFormatter, MaxNLocator
from plot_style import ACCENT_RED, BLACK, CATEGORICAL, GREY, MM_TO_INCH, save_pdf_figure


def save_cns_figure(fig, filename):
    """Save a vector PDF figure for direct use in LaTeX."""
    save_pdf_figure(fig, filename)


def summarize_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, reproducible data-quality table for the notebook."""
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_rate_pct": df.isna().mean().mul(100),
        "unique_values": df.nunique(dropna=False),
    })
    return summary.sort_values("missing_count", ascending=False)


def get_dataset_quality_metrics(df: pd.DataFrame) -> dict:
    """Return dataset-level quality checks without changing the input data."""
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [
            column for column in df.columns if df[column].nunique(dropna=False) <= 1
        ],
    }


def plot_label_distribution(
    train_df: pd.DataFrame,
    output_dir="image/数据质量和标签分布",
) -> Path:
    """Plot the highly imbalanced fraud label distribution."""
    counts = train_df["is_fraud"].value_counts().reindex([0, 1], fill_value=0)
    percentages = counts.div(counts.sum()).mul(100)
    labels = ["Legitimate", "Fraud"]
    colors = [CATEGORICAL[0], ACCENT_RED]

    fig, ax = plt.subplots(figsize=(89 * MM_TO_INCH, 72 * MM_TO_INCH))
    bars = ax.bar(labels, counts.to_numpy(), color=colors, width=0.58)
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_xlabel("Transaction class")
    ax.set_title("Fraud is rare", pad=10, fontweight="bold")
    ax.set_ylim(bottom=1, top=max(counts) * 20)
    ax.set_yticks([1, 100, 10_000, 1_000_000], ["1", "100", "10,000", "1M"])
    ax.minorticks_off()

    for bar, count, percentage in zip(bars, counts, percentages):
        ax.annotate(
            f"{count:,}\n({percentage:.4f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, count),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=BLACK,
        )

    fig.tight_layout(pad=1.2)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / "label_distribution"
    save_cns_figure(fig, filename)
    plt.close(fig)
    return filename.with_suffix(".pdf")


def plot_time_distribution(
    train_df: pd.DataFrame,
    output_dir="image/时间分布",
) -> Path:
    """Plot daily transaction volume and fraud concentration over elapsed time."""
    daily_input = train_df.assign(day_index=train_df["hours_elapsed"] // 24)
    daily = (
        daily_input.groupby("day_index", as_index=True)
        .agg(transaction_count=("hours_elapsed", "size"), fraud_count=("is_fraud", "sum"))
        .sort_index()
    )
    daily["fraud_rate_pct"] = daily["fraud_count"].div(daily["transaction_count"]).mul(100)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(89 * MM_TO_INCH, 110 * MM_TO_INCH),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15]},
    )
    x = daily.index.to_numpy()

    axes[0].plot(x, daily["transaction_count"], color=CATEGORICAL[0], linewidth=1.5,
                 marker="o", markersize=2.8)
    axes[0].fill_between(x, daily["transaction_count"], color=CATEGORICAL[0], alpha=0.12)
    axes[0].set_ylabel("Transactions (millions)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}"))
    axes[0].yaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_title("Transactions over time",
                      pad=10, fontweight="bold")

    axes[1].bar(x, daily["fraud_count"], color=ACCENT_RED, alpha=0.58, width=0.8,
                label="Fraud count")
    axes[1].set_xlabel("Day index (24-hour bins)")
    axes[1].set_ylabel("Fraud count")
    axes[1].set_ylim(0, daily["fraud_count"].max() * 1.25)
    axes[1].yaxis.set_major_locator(MaxNLocator(4))
    axes[1].set_xticks([0, 10, 20, 30])

    fraud_axis = axes[1].twinx()
    fraud_axis.plot(x, daily["fraud_rate_pct"], color=BLACK, linewidth=1.2,
                    marker="o", markersize=3.2, label="Fraud rate")
    fraud_axis.set_ylabel("Fraud rate (%)")
    fraud_axis.spines["top"].set_visible(False)

    first_fraud_day = daily.index[daily["fraud_count"] > 0].min()
    axes[1].axvline(first_fraud_day, color=GREY, linestyle="--", linewidth=0.8)
    axes[1].annotate(
        f"First fraud observed\n(day {first_fraud_day})",
        xy=(first_fraud_day, daily.loc[first_fraud_day, "fraud_count"]),
        xytext=(0.04, 0.58),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.8},
    )

    handles_a, labels_a = axes[1].get_legend_handles_labels()
    handles_b, labels_b = fraud_axis.get_legend_handles_labels()
    axes[1].legend(handles_a + handles_b, labels_a + labels_b,
                   loc="upper left", bbox_to_anchor=(-0.12, 1.30), ncol=2,
                   handlelength=1.2, columnspacing=0.8)

    fig.tight_layout(pad=1.2)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / "time_distribution_training"
    save_cns_figure(fig, filename)
    plt.close(fig)
    return filename.with_suffix(".pdf")
