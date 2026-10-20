"""Evaluate the supplied fraud flag as a standalone reference classifier."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from plot_style import ACCENT_RED, BLACK, CATEGORICAL, MM_TO_INCH, save_pdf_figure


def summarize_system_flag(train_df: pd.DataFrame) -> dict:
    """Evaluate is_flagged_fraud as a pre-existing binary classifier."""
    confusion = pd.crosstab(
        train_df["is_fraud"],
        train_df["is_flagged_fraud"],
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    true_negative = int(confusion.loc[0, 0])
    false_positive = int(confusion.loc[0, 1])
    false_negative = int(confusion.loc[1, 0])
    true_positive = int(confusion.loc[1, 1])
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    metrics = pd.Series(
        {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        name="value",
    )
    return {"confusion_matrix": confusion, "metrics": metrics}


def plot_system_flag_performance(
    train_df: pd.DataFrame,
    output_dir="image/system_flag_performance",
) -> Path:
    """Plot the legacy system confusion matrix and fraud-class metrics."""
    result = summarize_system_flag(train_df)
    confusion = result["confusion_matrix"]
    metrics = result["metrics"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(89 * MM_TO_INCH, 120 * MM_TO_INCH),
        gridspec_kw={"height_ratios": [1.15, 0.85]},
    )

    cmap = LinearSegmentedColormap.from_list(
        "academic_blue",
        ["#F7FBFF", "#6BAED6", "#08306B"],
    )
    log_counts = np.log10(confusion.to_numpy(dtype=float) + 1)
    axes[0].imshow(log_counts, cmap=cmap, vmin=0, vmax=max(log_counts.max(), 1))
    axes[0].set_xticks([0, 1], ["Not flagged", "Flagged"])
    axes[0].set_yticks([0, 1], ["Legitimate", "Fraud"])
    axes[0].set_xlabel("System flag\nColor = log10(count + 1)")
    axes[0].set_ylabel("Actual label")
    axes[0].set_title("Existing flag: actual versus flagged", fontweight="bold")
    for row in range(2):
        for column in range(2):
            value = int(confusion.iloc[row, column])
            text_color = "white" if log_counts[row, column] > log_counts.max() * 0.55 else BLACK
            axes[0].text(
                column,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    labels = ["Precision", "Recall", "F1-score"]
    values = metrics.to_numpy() * 100
    bars = axes[1].bar(labels, values, color=[CATEGORICAL[0], ACCENT_RED, CATEGORICAL[2]],
                       width=0.58)
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title("Fraud-class metrics", fontweight="bold")
    axes[1].set_ylim(0, 115)
    for bar, value in zip(bars, values):
        axes[1].annotate(
            f"{value:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    fig.tight_layout(pad=1.2)
    return save_pdf_figure(fig, Path(output_dir) / "system_flag_performance")
