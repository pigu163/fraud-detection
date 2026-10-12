# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) transaction counts → assets/figures/BarCategorical/ → param inherit
# (b) fraud rates by category → assets/figures/BarCategorical/ → param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from plot_style import ACCENT_RED, BLACK, CATEGORICAL, MM_TO_INCH, save_pdf_figure


TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def summarize_transaction_types(train_df: pd.DataFrame) -> pd.DataFrame:
    """Return transaction counts, fraud counts, and fraud rates by type."""
    summary = (
        train_df.groupby("transaction_type")
        .agg(
            transaction_count=("is_fraud", "size"),
            fraud_count=("is_fraud", "sum"),
            flagged_count=("is_flagged_fraud", "sum"),
        )
        .reindex(TRANSACTION_TYPES, fill_value=0)
    )
    summary["fraud_rate_pct"] = (
        summary["fraud_count"].div(summary["transaction_count"]).mul(100)
    )
    return summary


def plot_transaction_type_counts(
    train_df: pd.DataFrame,
    output_dir="image/交易类型与欺诈的关系",
) -> Path:
    """Plot the number of transactions for each transaction type."""
    summary = summarize_transaction_types(train_df)

    fig, ax = plt.subplots(figsize=(89 * MM_TO_INCH, 80 * MM_TO_INCH))
    bars = ax.barh(
        summary.index,
        summary["transaction_count"],
        color=CATEGORICAL[0],
        height=0.62,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Transactions (log scale)")
    ax.set_title("Volume by transaction type", pad=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(1_000, summary["transaction_count"].max() * 80)
    ax.set_xticks([1_000, 100_000, 10_000_000], ["1,000", "100,000", "10M"])
    ax.minorticks_off()

    for bar, value in zip(bars, summary["transaction_count"]):
        ax.annotate(
            f"{value:,}",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=BLACK,
        )

    fig.tight_layout(pad=1.2)
    return save_pdf_figure(fig, Path(output_dir) / "transaction_type_counts")


def plot_transaction_type_fraud_rates(
    train_df: pd.DataFrame,
    output_dir="image/交易类型与欺诈的关系",
) -> Path:
    """Plot fraud rates and annotate the number of fraud cases."""
    summary = summarize_transaction_types(train_df)

    fig, ax = plt.subplots(figsize=(89 * MM_TO_INCH, 80 * MM_TO_INCH))
    bars = ax.barh(
        summary.index,
        summary["fraud_rate_pct"],
        color=[ACCENT_RED if value > 0 else "#BFC5CC"
               for value in summary["fraud_rate_pct"]],
        height=0.62,
    )
    ax.set_xlabel("Fraud rate (%)")
    ax.set_title("Fraud rate by transaction type", pad=10, fontweight="bold")
    ax.invert_yaxis()
    top = max(summary["fraud_rate_pct"].max() * 2.1, 0.065)
    ax.set_xlim(0, top)
    ax.set_xticks([0, 0.02, 0.04, 0.06])

    for bar, rate, count in zip(
        bars, summary["fraud_rate_pct"], summary["fraud_count"]
    ):
        ax.annotate(
            f"{rate:.4f}%\n{count:,} fraud",
            xy=(rate, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=BLACK,
        )

    fig.tight_layout(pad=1.2)
    return save_pdf_figure(fig, Path(output_dir) / "transaction_type_fraud_rates")
