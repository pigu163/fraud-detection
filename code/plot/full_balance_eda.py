"""Training-only EDA of amount relative to the initiator's starting balance.

The rule uses only fields available before a transaction completes. These
summaries describe the full training file; they are not validation metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import ACCENT_RED, BLACK, CATEGORICAL, GREY, MM_TO_INCH, save_pdf_figure


FRAUD_TYPES = ("CASH_OUT", "TRANSFER")
RATIO_BINS = (
    "Zero starting balance",
    "Below 25%",
    "25% to <75%",
    "75% to <100%",
    "Full balance ±$0.01",
    ">100% to 125%",
    "Above 125%",
)
TIME_EDGES = (634, 646, 658, 670, 682, 694, 706, 718, 743)
TIME_LABELS = tuple(
    f"{left + 1}–{right}" for left, right in zip(TIME_EDGES[:-1], TIME_EDGES[1:])
)


def full_balance_rule(df: pd.DataFrame) -> pd.Series:
    """Positive starting balance and amount within one cent of that balance."""
    before = df["initiater_balance_before"]
    amount = df["transaction_amount"]
    return (before > 0) & ((amount - before).abs() <= 0.01)


def summarize_full_balance_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and fraud rates by transaction type and rule status."""
    subset = df.loc[df["transaction_type"].isin(FRAUD_TYPES)]
    result = (
        subset.assign(rule_match=full_balance_rule(subset))
        .groupby(["transaction_type", "rule_match"], observed=True)
        .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
        .reindex(pd.MultiIndex.from_product([FRAUD_TYPES, [True, False]], names=["transaction_type", "rule_match"]), fill_value=0)
    )
    result["fraud_rate_pct"] = result["fraud"].div(result["transactions"]).mul(100)
    return result


def summarize_balance_ratio_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Exclusive ratio bins, with zero denominator and the exact rule separate."""
    subset = df.loc[df["transaction_type"].isin(FRAUD_TYPES)]
    before = subset["initiater_balance_before"].to_numpy(dtype=float)
    amount = subset["transaction_amount"].to_numpy(dtype=float)
    if np.any(before < 0):
        raise ValueError("Negative starting balances need a separate ratio category")
    ratio = np.divide(amount, before, out=np.full(len(subset), np.nan), where=before > 0)
    rule = full_balance_rule(subset).to_numpy()
    categories = np.select(
        [
            before == 0,
            rule,
            ratio < 0.25,
            ratio < 0.75,
            ratio < 1,
            ratio <= 1.25,
        ],
        [
            RATIO_BINS[0], RATIO_BINS[4], RATIO_BINS[1],
            RATIO_BINS[2], RATIO_BINS[3], RATIO_BINS[5],
        ],
        default=RATIO_BINS[6],
    )
    result = (
        subset.assign(ratio_bin=pd.Categorical(categories, categories=RATIO_BINS, ordered=True))
        .groupby(["transaction_type", "ratio_bin"], observed=False)
        .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
    )
    result["fraud_rate_pct"] = result["fraud"].div(result["transactions"]).mul(100)
    return result


def summarize_rule_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Rule hits, missed fraud and legitimate counts in explicit hour windows."""
    subset = df.loc[
        df["transaction_type"].isin(FRAUD_TYPES)
        & df["hours_elapsed"].between(TIME_EDGES[0] + 1, TIME_EDGES[-1])
    ]
    periods = pd.cut(
        subset["hours_elapsed"], bins=TIME_EDGES, labels=TIME_LABELS,
        include_lowest=False, right=True,
    )
    result = (
        subset.assign(
            period=periods,
            matched_fraud=full_balance_rule(subset) & subset["is_fraud"].eq(1),
        )
        .groupby(["transaction_type", "period"], observed=False)
        .agg(
            transactions=("is_fraud", "size"),
            fraud=("is_fraud", "sum"),
            rule_hits=("matched_fraud", "sum"),
        )
        .reindex(pd.MultiIndex.from_product([FRAUD_TYPES, TIME_LABELS], names=["transaction_type", "period"]), fill_value=0)
    )
    result["missed_fraud"] = result["fraud"] - result["rule_hits"]
    result["legitimate"] = result["transactions"] - result["fraud"]
    return result


def plot_full_balance_rule(df: pd.DataFrame, output_dir="image/交易金额与余额行为") -> Path:
    """Compare exact rule matches with all other rows within each fraud type."""
    summary = summarize_full_balance_rule(df)
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM_TO_INCH, 78 * MM_TO_INCH))
    for ax, transaction_type in zip(axes, FRAUD_TYPES):
        block = summary.loc[transaction_type]
        y = np.arange(2)
        counts = block["transactions"].to_numpy()
        ax.barh(y, counts, color=[ACCENT_RED, CATEGORICAL[0]], height=0.54)
        ax.set_yticks(y, ["Rule match", "Other"])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(1, counts.max() * 20)
        ax.set_xticks([1, 1_000, 1_000_000], ["1", "1k", "1M"])
        ax.minorticks_off()
        ax.set_xlabel("Transactions (log scale)")
        ax.set_title(transaction_type, fontweight="bold")
        for yi, (_, row) in enumerate(block.iterrows()):
            rate = row["fraud_rate_pct"]
            rate_label = "100%" if rate == 100 else f"{rate:.4f}%"
            ax.annotate(
                f"{int(row['fraud']):,}/{int(row['transactions']):,} fraud ({rate_label})",
                xy=(row["transactions"], yi), xytext=(5, 0),
                textcoords="offset points", va="center", color=BLACK,
            )
    fig.suptitle("Positive-balance full withdrawal: rule versus other transactions", fontweight="bold")
    return save_pdf_figure(fig, Path(output_dir) / "full_balance_rule_by_type")


def plot_balance_ratio_bins(df: pd.DataFrame, output_dir="image/交易金额与余额行为") -> Path:
    """Show bin support, fraud count and rate without compressing small rates."""
    summary = summarize_balance_ratio_bins(df)
    fig, axes = plt.subplots(2, 1, figsize=(183 * MM_TO_INCH, 110 * MM_TO_INCH))
    for ax, transaction_type in zip(axes, FRAUD_TYPES):
        block = summary.loc[transaction_type]
        y = np.arange(len(block))
        counts = block["transactions"].to_numpy()
        colors = [ACCENT_RED if name == RATIO_BINS[4] else CATEGORICAL[0] for name in block.index]
        ax.barh(y, counts, color=colors, height=0.65)
        ax.set_yticks(y, block.index)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(1, max(counts.max() * 1.2, 10))
        ax.set_xticks([1, 100, 10_000, 100_000], ["1", "100", "10k", "100k"])
        ax.minorticks_off()
        ax.set_xlabel("Transactions (log scale)")
        ax.set_title(transaction_type, fontweight="bold")
        for yi, (_, row) in enumerate(block.iterrows()):
            if row["transactions"]:
                rate = row["fraud_rate_pct"]
                rate_label = "100%" if rate == 100 else f"{rate:.4f}%"
                annotation = f"n={int(row['transactions']):,}; fraud={int(row['fraud']):,}; {rate_label}"
            else:
                annotation = "No observations"
            ax.text(1.02, yi, annotation, transform=ax.get_yaxis_transform(), va="center", color=BLACK)
    fig.suptitle("Transaction amount / initiator starting balance", fontweight="bold")
    return save_pdf_figure(fig, Path(output_dir) / "amount_to_starting_balance_bins")


def plot_rule_over_time(df: pd.DataFrame, output_dir="image/时间分布") -> Path:
    """Separate fraud rule hits from denominators over time and by type."""
    summary = summarize_rule_over_time(df)
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM_TO_INCH, 130 * MM_TO_INCH), sharex="col")
    x = np.arange(len(TIME_LABELS))
    for col, transaction_type in enumerate(FRAUD_TYPES):
        block = summary.loc[transaction_type]
        hits = block["rule_hits"].to_numpy()
        missed = block["missed_fraud"].to_numpy()
        legitimate = block["legitimate"].to_numpy()
        axes[0, col].bar(x, hits, color=ACCENT_RED, label="Rule hits")
        axes[0, col].bar(x, missed, bottom=hits, color=GREY, label="Missed fraud")
        axes[0, col].set_title(transaction_type, fontweight="bold")
        axes[0, col].set_ylabel("Fraud transactions")
        axes[0, col].set_ylim(0, max((hits + missed).max() * 1.3, 1))
        axes[1, col].bar(x, legitimate, color=CATEGORICAL[0])
        axes[1, col].set_ylabel("Legitimate transactions")
        axes[1, col].set_xlabel("Elapsed-hour window")
        axes[1, col].set_xticks(x, TIME_LABELS, rotation=45, ha="right")
        axes[1, col].set_ylim(0, max(legitimate.max() * 1.3, 1))
        for xi, count in enumerate(legitimate):
            if count == 0:
                axes[1, col].annotate("0", (xi, 0), xytext=(0, 4), textcoords="offset points", ha="center")
    axes[0, 0].legend(loc="upper right", ncol=2)
    fig.suptitle("Full-balance rule over time (hours 635–743)", fontweight="bold")
    return save_pdf_figure(fig, Path(output_dir) / "full_balance_rule_over_time")
