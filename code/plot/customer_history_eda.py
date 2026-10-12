# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) account-role composition → assets/figures/BarCategorical/ → param inherit
# (b) account activity distribution → assets/figures/LineTrend/ → param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import ACCENT_RED, BLACK, CATEGORICAL, MM_TO_INCH, save_pdf_figure


ROLE_PAIRS = ["C→C", "C→M", "M→C", "M→M"]


def _role_pair_series(train_df: pd.DataFrame) -> pd.Series:
    """Build role-pair labels from the four binary customer-type columns."""
    if not {
        "initiating_customer_is_C",
        "initiating_customer_is_M",
        "recipient_customer_is_C",
        "recipient_customer_is_M",
    }.issubset(train_df.columns):
        raise KeyError("Run the customer-type feature block before this analysis.")

    return pd.Series(
        np.select(
            [
                (train_df["initiating_customer_is_C"] == 1)
                & (train_df["recipient_customer_is_C"] == 1),
                (train_df["initiating_customer_is_C"] == 1)
                & (train_df["recipient_customer_is_M"] == 1),
                (train_df["initiating_customer_is_M"] == 1)
                & (train_df["recipient_customer_is_C"] == 1),
                (train_df["initiating_customer_is_M"] == 1)
                & (train_df["recipient_customer_is_M"] == 1),
            ],
            ROLE_PAIRS,
            default="Other",
        ),
        index=train_df.index,
        name="role_pair",
    )


def _strict_prior_hour_table(
    train_df: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """Count transactions from strictly earlier hours for each entity."""
    entity_hour = (
        train_df.groupby([entity_column, "hours_elapsed"])
        .size()
        .rename("transactions_in_hour")
        .reset_index()
        .sort_values([entity_column, "hours_elapsed"], kind="stable")
    )
    cumulative = entity_hour.groupby(entity_column)["transactions_in_hour"].cumsum()
    entity_hour["prior_transactions_strict_hour"] = (
        cumulative - entity_hour["transactions_in_hour"]
    )
    return entity_hour


def summarize_customer_history(train_df: pd.DataFrame) -> dict:
    """Summarize role pairs, repeated accounts, and strict historical coverage."""
    role_pairs = _role_pair_series(train_df)
    role_summary = (
        train_df.assign(_role_pair=role_pairs)
        .groupby("_role_pair")
        .agg(transaction_count=("is_fraud", "size"), fraud_count=("is_fraud", "sum"))
        .reindex(ROLE_PAIRS, fill_value=0)
    )
    role_summary["fraud_rate_pct"] = (
        role_summary["fraud_count"].div(role_summary["transaction_count"]).fillna(0).mul(100)
    )

    initiator_activity = train_df.groupby("initiating_customer").agg(
        transaction_count=("is_fraud", "size"),
        active_hours=("hours_elapsed", "nunique"),
        unique_recipients=("recipient_customer", "nunique"),
        total_amount=("transaction_amount", "sum"),
    )
    recipient_activity = train_df.groupby("recipient_customer").agg(
        transaction_count=("is_fraud", "size"),
        active_hours=("hours_elapsed", "nunique"),
        unique_initiators=("initiating_customer", "nunique"),
        total_amount=("transaction_amount", "sum"),
    )

    initiator_set = set(train_df["initiating_customer"].dropna())
    recipient_set = set(train_df["recipient_customer"].dropna())
    initiator_prior = _strict_prior_hour_table(train_df, "initiating_customer")
    recipient_prior = _strict_prior_hour_table(train_df, "recipient_customer")
    initiator_rows_with_history = int(
        initiator_prior.loc[
            initiator_prior["prior_transactions_strict_hour"] > 0,
            "transactions_in_hour",
        ].sum()
    )
    recipient_rows_with_history = int(
        recipient_prior.loc[
            recipient_prior["prior_transactions_strict_hour"] > 0,
            "transactions_in_hour",
        ].sum()
    )

    history_metrics = pd.Series(
        {
            "unique_initiating_accounts": len(initiator_activity),
            "repeated_initiating_accounts": int(
                (initiator_activity["transaction_count"] > 1).sum()
            ),
            "unique_recipient_accounts": len(recipient_activity),
            "repeated_recipient_accounts": int(
                (recipient_activity["transaction_count"] > 1).sum()
            ),
            "accounts_in_both_roles": len(initiator_set & recipient_set),
            "initiators_active_in_multiple_hours": int(
                (initiator_activity["active_hours"] > 1).sum()
            ),
            "recipients_active_in_multiple_hours": int(
                (recipient_activity["active_hours"] > 1).sum()
            ),
            "transactions_with_strict_prior_initiator_history": initiator_rows_with_history,
            "transactions_with_strict_prior_recipient_history": recipient_rows_with_history,
        },
        name="value",
    )
    return {
        "role_summary": role_summary,
        "history_metrics": history_metrics,
        "initiator_activity": initiator_activity,
        "recipient_activity": recipient_activity,
        "initiator_prior_by_hour": initiator_prior,
        "recipient_prior_by_hour": recipient_prior,
    }


def _activity_bins(activity: pd.Series) -> pd.Series:
    """Aggregate account counts into readable activity bins."""
    bins = [0, 1, 2, 3, 5, 10, np.inf]
    labels = ["1", "2", "3", "4–5", "6–10", ">10"]
    return pd.cut(activity, bins=bins, labels=labels, include_lowest=True)


def plot_customer_history(
    train_df: pd.DataFrame,
    output_dir="image/客户类型与账户历史",
) -> Path:
    """Plot customer-role composition and account-history availability."""
    summary = summarize_customer_history(train_df)
    role_summary = summary["role_summary"]
    initiator_activity = summary["initiator_activity"]
    recipient_activity = summary["recipient_activity"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(183 * MM_TO_INCH, 132 * MM_TO_INCH),
    )
    x = np.arange(len(ROLE_PAIRS))

    bars = axes[0, 0].bar(
        ROLE_PAIRS,
        role_summary["transaction_count"],
        color=CATEGORICAL[0],
        width=0.62,
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Initiator → recipient type")
    axes[0, 0].set_ylabel("Transactions (log scale)")
    axes[0, 0].set_title("Volume by account pair", fontweight="bold")
    axes[0, 0].set_yticks([1, 100, 10_000, 1_000_000], ["1", "100", "10,000", "1M"])
    axes[0, 0].minorticks_off()
    axes[0, 0].set_ylim(bottom=1, top=max(role_summary["transaction_count"].max(), 1) * 3)
    for bar, value in zip(bars, role_summary["transaction_count"]):
        axes[0, 0].annotate(
            f"{value:,}",
            xy=(bar.get_x() + bar.get_width() / 2, max(value, 1)),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    bars = axes[0, 1].bar(
        ROLE_PAIRS,
        role_summary["fraud_rate_pct"],
        color=[ACCENT_RED if value > 0 else "#BFC5CC"
               for value in role_summary["fraud_rate_pct"]],
        width=0.62,
    )
    axes[0, 1].set_xlabel("Initiator → recipient type")
    axes[0, 1].set_ylabel("Fraud rate (%)")
    axes[0, 1].set_title("Fraud rate by account pair", fontweight="bold")
    axes[0, 1].set_ylim(0, max(role_summary["fraud_rate_pct"].max() * 1.45, 0.05))
    for bar, rate, count, total in zip(
        bars, role_summary["fraud_rate_pct"], role_summary["fraud_count"],
        role_summary["transaction_count"],
    ):
        axes[0, 1].annotate(
            f"{rate:.4f}%\n(n={count:,})" if total else "No data",
            xy=(bar.get_x() + bar.get_width() / 2, rate),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    bins = ["1", "2", "3", "4–5", "6–10", ">10"]
    initiator_bin_counts = _activity_bins(initiator_activity["transaction_count"]).value_counts()
    recipient_bin_counts = _activity_bins(recipient_activity["transaction_count"]).value_counts()
    axes[1, 0].bar(
        np.arange(len(bins)) - 0.18,
        initiator_bin_counts.reindex(bins, fill_value=0),
        width=0.36,
        color=CATEGORICAL[0],
        label="Initiating accounts",
    )
    axes[1, 0].bar(
        np.arange(len(bins)) + 0.18,
        recipient_bin_counts.reindex(bins, fill_value=0),
        width=0.36,
        color=CATEGORICAL[2],
        label="Recipient accounts",
    )
    axes[1, 0].set_xticks(np.arange(len(bins)), bins)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Transactions observed per account")
    axes[1, 0].set_ylabel("Accounts (log scale)")
    axes[1, 0].set_yticks([1, 100, 10_000, 1_000_000], ["1", "100", "10,000", "1M"])
    axes[1, 0].minorticks_off()
    axes[1, 0].set_title("Account activity distribution", fontweight="bold")
    axes[1, 0].legend(loc="upper right", fontsize=7)

    activity_metrics = pd.DataFrame(
        {
            "Initiating accounts": [
                (initiator_activity["transaction_count"] > 1).mean() * 100,
                (initiator_activity["active_hours"] > 1).mean() * 100,
            ],
            "Recipient accounts": [
                (recipient_activity["transaction_count"] > 1).mean() * 100,
                (recipient_activity["active_hours"] > 1).mean() * 100,
            ],
        },
        index=["Repeated\n(2+ transactions)", "Active in\n2+ hours"],
    )
    activity_metrics.plot(
        kind="bar",
        ax=axes[1, 1],
        color=[CATEGORICAL[0], CATEGORICAL[2]],
        width=0.68,
    )
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Accounts (%)")
    axes[1, 1].set_title("Accounts with repeat activity", fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=0)
    axes[1, 1].legend(loc="upper right", fontsize=7)
    axes[1, 1].set_ylim(0, max(activity_metrics.to_numpy().max() * 1.35, 1))

    fig.suptitle("Customer types and account history", y=1.01,
                 fontsize=11, fontweight="bold")
    fig.tight_layout(pad=1.2)
    return save_pdf_figure(fig, Path(output_dir) / "customer_history")
