"""Training-only summaries and figures for transaction amounts and balances."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import ACCENT_RED, BLACK, CATEGORICAL, MM_TO_INCH, save_pdf_figure


TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def summarize_amounts(train_df: pd.DataFrame) -> pd.DataFrame:
    """Return amount quantiles and zero-amount rates by fraud label."""
    summary = train_df.groupby("is_fraud")["transaction_amount"].agg(
        transaction_count="size",
        mean="mean",
        median="median",
        q01=lambda values: values.quantile(0.01),
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
        q99=lambda values: values.quantile(0.99),
        zero_amount_rate_pct=lambda values: (values == 0).mean() * 100,
    )
    summary.index = ["Legitimate", "Fraud"]
    return summary


def summarize_balance_behavior(train_df: pd.DataFrame) -> dict:
    """Return balance diagnostics while keeping transaction types separate."""
    initiator_residual = (
        train_df["initiater_balance_before"]
        - train_df["transaction_amount"]
        - train_df["initiater_balance_after"]
    )
    recipient_residual = (
        train_df["recipient_balance_after"]
        - train_df["recipient_balance_before"]
        - train_df["transaction_amount"]
    )
    initiator_before = train_df["initiater_balance_before"].where(
        train_df["initiater_balance_before"] != 0
    )
    amount_to_initiator_before = train_df["transaction_amount"].div(initiator_before)
    tolerance = 0.01
    by_type = pd.DataFrame(index=TRANSACTION_TYPES)
    by_type["median_amount_to_initiator_before_pct"] = (
        amount_to_initiator_before.groupby(train_df["transaction_type"])
        .median()
        .mul(100)
        .reindex(TRANSACTION_TYPES)
    )
    by_type["initiator_after_zero_rate_pct"] = (
        train_df.groupby("transaction_type")["initiater_balance_after"]
        .apply(lambda values: (values == 0).mean() * 100)
        .reindex(TRANSACTION_TYPES)
    )
    by_type["recipient_after_zero_rate_pct"] = (
        train_df.groupby("transaction_type")["recipient_balance_after"]
        .apply(lambda values: (values == 0).mean() * 100)
        .reindex(TRANSACTION_TYPES)
    )
    by_type["initiator_balance_consistency_pct"] = (
        train_df.assign(_residual=initiator_residual)
        .groupby("transaction_type")["_residual"]
        .apply(lambda values: (values.abs() <= tolerance).mean() * 100)
        .reindex(TRANSACTION_TYPES)
    )
    by_type["recipient_balance_consistency_pct"] = (
        train_df.assign(_residual=recipient_residual)
        .groupby("transaction_type")["_residual"]
        .apply(lambda values: (values.abs() <= tolerance).mean() * 100)
        .reindex(TRANSACTION_TYPES)
    )
    return {
        "by_transaction_type": by_type,
        "amount_to_initiator_before": amount_to_initiator_before,
        "initiator_residual": initiator_residual,
        "recipient_residual": recipient_residual,
    }


def plot_amount_balance_behavior(
    train_df: pd.DataFrame,
    output_dir="image/amount_and_balance_behavior",
) -> Path:
    """Plot amount distributions, zero amounts, and balance diagnostics."""
    amount_summary = summarize_amounts(train_df)
    balance_summary = summarize_balance_behavior(train_df)["by_transaction_type"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(183 * MM_TO_INCH, 132 * MM_TO_INCH),
    )

    amount = train_df["transaction_amount"].to_numpy(dtype=float)
    log_amount = np.log1p(amount)
    bins = np.linspace(log_amount.min(), log_amount.max(), 70)
    for label, color, label_value in [
        ("Legitimate", CATEGORICAL[0], 0),
        ("Fraud", ACCENT_RED, 1),
    ]:
        values = log_amount[train_df["is_fraud"].to_numpy() == label_value]
        density, edges = np.histogram(values, bins=bins, density=True)
        axes[0, 0].step(
            edges[:-1],
            density,
            where="post",
            color=color,
            linewidth=1.5,
            label=label,
        )
        axes[0, 0].axvline(
            np.log1p(amount_summary.loc[label, "median"]),
            color=color,
            linestyle="--",
            linewidth=0.9,
        )
    axes[0, 0].set_xlabel("log1p(transaction amount)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Transaction amounts", fontweight="bold")
    axes[0, 0].legend(loc="upper right")

    zero_rates = amount_summary["zero_amount_rate_pct"]
    bars = axes[0, 1].bar(
        zero_rates.index,
        zero_rates,
        color=[CATEGORICAL[0], ACCENT_RED],
        width=0.58,
    )
    axes[0, 1].set_ylabel("Zero-amount transactions (%)")
    axes[0, 1].set_title("Zero-amount rate", fontweight="bold")
    axes[0, 1].set_ylim(0, max(zero_rates.max() * 1.35, 1))
    for bar, value in zip(bars, zero_rates):
        axes[0, 1].annotate(
            f"{value:.3f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    x = np.arange(len(TRANSACTION_TYPES))
    width = 0.36
    bars_a = axes[1, 0].bar(
        x - width / 2,
        balance_summary["initiator_after_zero_rate_pct"],
        width,
        label="Initiator after balance",
        color=CATEGORICAL[0],
    )
    bars_b = axes[1, 0].bar(
        x + width / 2,
        balance_summary["recipient_after_zero_rate_pct"],
        width,
        label="Recipient after balance",
        color=CATEGORICAL[2],
    )
    axes[1, 0].set_xticks(x, TRANSACTION_TYPES, rotation=25)
    axes[1, 0].set_ylabel("Zero after-balance rate (%)")
    axes[1, 0].set_title("Zero after-balance rate by type", fontweight="bold")
    axes[1, 0].set_ylim(0, 115)
    axes[1, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), fontsize=9)
    for bars_group in [bars_a, bars_b]:
        for bar in bars_group:
            if bar.get_height() > 0:
                axes[1, 0].annotate(
                    f"{bar.get_height():.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )

    bars_c = axes[1, 1].bar(
        x - width / 2,
        balance_summary["initiator_balance_consistency_pct"],
        width,
        label="Initiator balance",
        color=CATEGORICAL[0],
    )
    bars_d = axes[1, 1].bar(
        x + width / 2,
        balance_summary["recipient_balance_consistency_pct"],
        width,
        label="Recipient balance",
        color=CATEGORICAL[2],
    )
    axes[1, 1].set_xticks(x, TRANSACTION_TYPES, rotation=25)
    axes[1, 1].set_ylabel("Balance-consistency rate (%)")
    axes[1, 1].set_title("Balance arithmetic consistency", fontweight="bold")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), fontsize=9)
    fig.text(
        0.5,
        -0.045,
        "Consistency: absolute balance residual at most $0.01 (by transaction type).",
        ha="center",
        fontsize=9,
        color=BLACK,
    )

    fig.suptitle("Transaction amounts and balance behavior", y=1.01,
                 fontsize=11, fontweight="bold")
    fig.tight_layout(pad=1.2)
    return save_pdf_figure(
        fig,
        Path(output_dir) / "amount_balance_behavior",
    )
