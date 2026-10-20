"""Plots and summaries for the training-data quality section."""

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    ACCENT_RED,
    BLACK,
    CATEGORICAL,
    GREY,
    MM_TO_INCH,
    save_pdf_figure,
)


NUMERIC_ZERO_COLUMNS = [
    "transaction_amount",
    "initiater_balance_before",
    "initiater_balance_after",
    "recipient_balance_before",
    "recipient_balance_after",
]

BALANCE_ZERO_COLUMNS = NUMERIC_ZERO_COLUMNS[1:]

_ROLE_BY_COLUMN = {
    "initiating_customer": "Customer ID",
    "recipient_customer": "Customer ID",
    "transaction_type": "Category",
    "hours_elapsed": "Time",
    "is_fraud": "Target",
    "is_flagged_fraud": "System flag",
}

_ROLE_COLORS = {
    "Customer ID": CATEGORICAL[0],
    "Category": CATEGORICAL[2],
    "Time": CATEGORICAL[3],
    "Target": ACCENT_RED,
    "System flag": GREY,
    "Numeric": CATEGORICAL[4],
}


def summarize_zero_rates(
    df: pd.DataFrame,
    label_column: str = "is_fraud",
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return exact-zero counts and rates for numeric fields by class.

    A zero is treated as an observed numeric value, not as a missing value.
    The result is computed from the training data passed by the notebook.
    """
    columns = list(columns or NUMERIC_ZERO_COLUMNS)
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing numeric columns: {missing_columns}")
    if label_column not in df.columns:
        raise KeyError(f"Missing label column: {label_column}")

    rows = []
    for label, group in df.groupby(label_column, sort=True, dropna=False):
        label_name = "Fraud" if label == 1 else "Legitimate"
        for column in columns:
            zero_count = int(group[column].eq(0).sum())
            rows.append(
                {
                    "label": int(label),
                    "label_name": label_name,
                    "column": column,
                    "zero_count": zero_count,
                    "row_count": len(group),
                    "zero_rate_pct": zero_count / len(group) * 100,
                }
            )
    return pd.DataFrame(rows)


def plot_zero_value_rates(
    df: pd.DataFrame,
    output_dir: str = "image/data_quality_and_labels",
) -> Path:
    """Compare exact-zero rates for amount and balances by fraud label."""
    summary = summarize_zero_rates(df)
    display_names = {
        "transaction_amount": "Transaction amount",
        "initiater_balance_before": "Initiator\nbefore",
        "initiater_balance_after": "Initiator\nafter",
        "recipient_balance_before": "Recipient\nbefore",
        "recipient_balance_after": "Recipient\nafter",
    }
    colors = {"Legitimate": CATEGORICAL[0], "Fraud": ACCENT_RED}
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 * MM_TO_INCH, 82 * MM_TO_INCH),
        gridspec_kw={"width_ratios": [0.78, 1.55]},
    )

    for ax, columns, title in [
        (axes[0], ["transaction_amount"], "Transaction amount"),
        (axes[1], BALANCE_ZERO_COLUMNS, "Balance fields"),
    ]:
        positions = np.arange(len(columns))
        width = 0.34
        for offset, label_name in [(-width / 2, "Legitimate"), (width / 2, "Fraud")]:
            values = (
                summary[summary["label_name"] == label_name]
                .set_index("column")
                .reindex(columns)["zero_rate_pct"]
                .to_numpy()
            )
            bars = ax.bar(
                positions + offset,
                values,
                width=width,
                color=colors[label_name],
                label=label_name,
            )
            for bar, value in zip(bars, values):
                ax.annotate(
                    f"{value:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=BLACK,
                )
        ax.set_title(title, pad=9, fontweight="bold")
        ax.set_ylabel("Exact-zero rate (%)")
        ax.set_xticks(positions)
        ax.set_xticklabels([display_names[column] for column in columns])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.suptitle(
        "Exact-zero values by transaction label",
        y=1.02,
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(pad=1.2)
    return save_pdf_figure(fig, Path(output_dir) / "zero_value_rates")


def summarize_field_cardinality(
    df: pd.DataFrame,
    engineered_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Summarize unique values for raw fields, excluding engineered columns."""
    excluded = set(engineered_columns or [])
    columns = [column for column in df.columns if column not in excluded]
    rows = []
    for column in columns:
        unique_count = int(df[column].nunique(dropna=False))
        rows.append(
            {
                "column": column,
                "role": _ROLE_BY_COLUMN.get(column, "Numeric"),
                "dtype": str(df[column].dtype),
                "unique_count": unique_count,
                "unique_rate_pct": unique_count / len(df) * 100,
                "missing_count": int(df[column].isna().sum()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["unique_count", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )


def plot_field_cardinality(
    df: pd.DataFrame,
    output_dir: str = "image/data_quality_and_labels",
    engineered_columns: Optional[Iterable[str]] = None,
) -> Path:
    """Show raw-field cardinality as counts and as a fraction of all rows."""
    if engineered_columns is None:
        engineered_columns = [
            column for column in df.columns
            if column.startswith(("initiating_customer_is_", "recipient_customer_is_"))
        ]
    summary = summarize_field_cardinality(df, engineered_columns=engineered_columns)
    y = np.arange(len(summary))
    colors = [_ROLE_COLORS[row_role] for row_role in summary["role"]]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 * MM_TO_INCH, 94 * MM_TO_INCH),
        sharey=True,
        gridspec_kw={"width_ratios": [1.1, 1]},
    )

    axes[0].barh(y, summary["unique_count"], color=colors, height=0.68)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Unique values (log scale)")
    axes[0].set_title("Unique-value count", pad=9, fontweight="bold")
    axes[0].set_yticks(y)
    display_names = {
        "transaction_amount": "Amount",
        "initiating_customer": "Initiator ID",
        "recipient_customer": "Recipient ID",
        "initiater_balance_before": "Initiator balance before",
        "initiater_balance_after": "Initiator balance after",
        "recipient_balance_before": "Recipient balance before",
        "recipient_balance_after": "Recipient balance after",
        "hours_elapsed": "Elapsed hours",
        "transaction_type": "Transaction type",
        "is_fraud": "Fraud label",
        "is_flagged_fraud": "System flag",
    }
    axes[0].set_yticklabels([display_names.get(c, c) for c in summary["column"]])
    axes[0].invert_yaxis()
    axes[0].set_xlim(1, summary["unique_count"].max() * 60)
    axes[0].set_xticks([1, 1_000, 1_000_000], ["1", "1,000", "1M"])
    axes[0].minorticks_off()

    max_count = summary["unique_count"].max()
    for yi, count in zip(y, summary["unique_count"]):
        axes[0].annotate(
            f"{count:,}",
            xy=(count, yi),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7,
            color=BLACK,
        )

    axes[1].barh(y, summary["unique_rate_pct"], color=colors, height=0.68)
    axes[1].set_xlabel("Unique values / all rows (%)")
    axes[1].set_title("Unique-value ratio", pad=9, fontweight="bold")
    axes[1].set_xlim(0, 120)
    axes[1].set_xticks([0, 50, 100])
    axes[1].tick_params(axis="y", labelleft=False)
    for yi, rate in zip(y, summary["unique_rate_pct"]):
        # Two or five distinct categories have a small nonzero ratio;
        # showing 0.00% would incorrectly suggest that no values exist.
        rate_label = "<0.01%" if 0 < rate < 0.01 else f"{rate:.2f}%"
        axes[1].annotate(
            rate_label,
            xy=(rate, yi),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7,
            color=BLACK,
        )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=_ROLE_COLORS[role])
        for role in ["Customer ID", "Category", "Time", "Target", "System flag", "Numeric"]
    ]
    axes[1].legend(
        legend_handles,
        ["Customer ID", "Category", "Time", "Target", "System flag", "Numeric"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        fontsize=7,
        ncol=3,
        handlelength=1,
        columnspacing=0.8,
    )
    fig.suptitle(
        "Raw-field cardinality in the training data",
        y=1.01,
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(pad=1.2)
    return save_pdf_figure(fig, Path(output_dir) / "field_cardinality")
