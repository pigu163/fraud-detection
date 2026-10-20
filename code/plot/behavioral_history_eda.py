"""Training-only EDA for short-term sender and recipient behaviour.

The features in this module use only transactions from strictly earlier
hours.  Transactions in the current hour are treated as a batch because the
dataset does not provide an ordering within an hour.  The module is intended
for descriptive EDA and does not read the external test file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "figure.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text


BLUE = "#2166AC"
RED = "#B2182B"
GREEN = "#1B7837"
GREY = "#777777"
BLACK = "#222222"
MM_TO_INCH = 1 / 25.4
GRAPH_TYPES = ("CASH_OUT", "TRANSFER")
COUNT_LABELS = ("0", "1", "2", "3–5", "6–10", "≥11")
COUNT_EDGES = (-0.5, 0.5, 1.5, 2.5, 5.5, 10.5, np.inf)
RATIO_LABELS = ("No history", "<1×", "1–2×", "2–5×", "≥5×")
RATIO_EDGES = (-np.inf, 0.0, 1.0, 2.0, 5.0, np.inf)
DORMANCY_LABELS = ("No history", "1h", "2–6h", "7–24h", "25–72h", ">72h")


def _save_pdf(fig: plt.Figure, path: str | Path) -> Path:
    """Save one report-ready PDF and never create a raster preview."""
    for artist in fig.findobj(match=Text):
        if artist.get_text():
            artist.set_fontsize(max(7, artist.get_fontsize()))
    fig.tight_layout(pad=1.0)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output.with_suffix(".pdf")


def _account_history(
    transactions: pd.DataFrame,
    account_column: str,
    prefix: str,
    windows: tuple[int, ...] = (24, 72),
) -> pd.DataFrame:
    """Create account history snapshots and map them back to transaction rows."""
    columns = [account_column, "hours_elapsed", "transaction_amount"]
    events = (
        transactions.loc[:, columns]
        .groupby([account_column, "hours_elapsed"], sort=False, as_index=False)
        .agg(
            current_count=("transaction_amount", "size"),
            current_amount_sum=("transaction_amount", "sum"),
            current_amount_max=("transaction_amount", "max"),
        )
        .sort_values([account_column, "hours_elapsed"], kind="stable")
        .reset_index(drop=True)
    )
    groups = events.groupby(account_column, sort=False)

    events["prior_count_all"] = (
        groups["current_count"].cumsum() - events["current_count"]
    )
    events["prior_amount_sum_all"] = (
        groups["current_amount_sum"].cumsum() - events["current_amount_sum"]
    )
    running_max = groups["current_amount_max"].cummax()
    events["prior_amount_max_all"] = running_max.groupby(
        events[account_column], sort=False
    ).shift(fill_value=0)
    events["prior_amount_mean_all"] = np.divide(
        events["prior_amount_sum_all"].to_numpy(dtype=float),
        events["prior_count_all"].to_numpy(dtype=float),
        out=np.zeros(len(events), dtype=float),
        where=events["prior_count_all"].to_numpy(dtype=float) > 0,
    )
    previous_hour = groups["hours_elapsed"].shift(1)
    events["time_since_last"] = (
        events["hours_elapsed"] - previous_hour
    ).where(previous_hour.notna())

    # Time-based rolling windows are evaluated on account-hour snapshots.
    # closed='left' excludes the current hour and therefore prevents leakage.
    events["_time"] = pd.to_timedelta(events["hours_elapsed"], unit="h")
    for window in windows:
        rolling = (
            events.set_index("_time")
            .groupby(account_column, sort=False)[
                ["current_count", "current_amount_sum"]
            ]
            .rolling(f"{window}h", closed="left")
            .sum()
            .reset_index()
            .rename(
                columns={
                    "current_count": f"{prefix}_count_{window}h",
                    "current_amount_sum": f"{prefix}_amount_sum_{window}h",
                }
            )
        )
        rolling["hours_elapsed"] = (
            rolling["_time"].dt.total_seconds() / 3600
        ).astype("int64")
        rolling = rolling.drop(columns="_time")
        events = events.merge(
            rolling,
            on=[account_column, "hours_elapsed"],
            how="left",
            sort=False,
            validate="one_to_one",
        )

    feature_names = {
        "prior_count_all": f"{prefix}_count_all",
        "prior_amount_sum_all": f"{prefix}_amount_sum_all",
        "prior_amount_mean_all": f"{prefix}_amount_mean_all",
        "prior_amount_max_all": f"{prefix}_amount_max_all",
        "time_since_last": f"{prefix}_time_since_last",
    }
    for window in windows:
        feature_names[f"{prefix}_count_{window}h"] = f"{prefix}_count_{window}h"
        feature_names[f"{prefix}_amount_sum_{window}h"] = (
            f"{prefix}_amount_sum_{window}h"
        )

    lookup = events.set_index([account_column, "hours_elapsed"])
    query = pd.MultiIndex.from_arrays(
        [
            transactions[account_column].to_numpy(),
            transactions["hours_elapsed"].to_numpy(),
        ],
        names=[account_column, "hours_elapsed"],
    )
    mapped = lookup.reindex(query)
    result = pd.DataFrame(index=transactions.index)
    for source_name, output_name in feature_names.items():
        values = mapped[source_name].to_numpy(dtype=float)
        # A missing rolling window means no earlier event in that window.
        # Keep it as a numeric zero so the zero-history bin is explicit.
        if "_count_" in output_name or "_amount_sum_" in output_name:
            values = np.nan_to_num(values, nan=0.0)
        result[output_name] = values
    result[f"{prefix}_has_prior_history"] = (
        result[f"{prefix}_count_all"] > 0
    ).astype("int8")
    result[f"{prefix}_amount_to_prior_mean"] = np.divide(
        transactions["transaction_amount"].to_numpy(dtype=float),
        result[f"{prefix}_amount_mean_all"].to_numpy(dtype=float),
        out=np.full(len(transactions), np.nan, dtype=float),
        where=result[f"{prefix}_amount_mean_all"].to_numpy(dtype=float) > 0,
    )
    return result


def build_behavioral_history_features(
    transactions: pd.DataFrame, windows: tuple[int, ...] = (24, 72)
) -> pd.DataFrame:
    """Build strictly prior-hour sender and recipient behavioural features."""
    required = {
        "hours_elapsed",
        "transaction_amount",
        "initiating_customer",
        "recipient_customer",
    }
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if transactions.empty:
        raise ValueError("Cannot build behavioural features from an empty dataset")
    work = transactions.reset_index(drop=True).copy()
    source = _account_history(
        work, "initiating_customer", "source_prior_out", windows=windows
    )
    target = _account_history(
        work, "recipient_customer", "target_prior_in", windows=windows
    )
    result = pd.concat([source, target], axis=1)
    return result


def _count_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(values, bins=COUNT_EDGES, labels=COUNT_LABELS)


def _ratio_bucket(values: pd.Series) -> pd.Series:
    bucket = pd.cut(values, bins=RATIO_EDGES, labels=RATIO_LABELS)
    return bucket.astype(object).where(values.notna(), "No history")


def _dormancy_bucket(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    result = pd.Series("No history", index=values.index, dtype="object")
    result.loc[values.eq(1)] = "1h"
    result.loc[values.between(2, 6, inclusive="both")] = "2–6h"
    result.loc[values.between(7, 24, inclusive="both")] = "7–24h"
    result.loc[values.between(25, 72, inclusive="both")] = "25–72h"
    result.loc[values.gt(72)] = ">72h"
    return result


def _summary(
    train_df: pd.DataFrame,
    features: pd.DataFrame,
    feature: str,
    bucket: pd.Series,
) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "feature": feature,
            "transaction_type": train_df["transaction_type"].to_numpy(),
            "bucket": bucket.to_numpy(),
            "is_fraud": train_df["is_fraud"].to_numpy(dtype=np.int64),
        }
    )
    grouped = (
        work.groupby(["feature", "transaction_type", "bucket"], observed=False)
        .agg(transaction_count=("is_fraud", "size"), fraud_count=("is_fraud", "sum"))
        .reset_index()
    )
    grouped["fraud_rate_pct"] = np.where(
        grouped["transaction_count"] > 0,
        100 * grouped["fraud_count"] / grouped["transaction_count"],
        np.nan,
    )
    return grouped


def summarize_behavioral_eda(
    train_df: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Return label summaries for the fixed, label-independent bins."""
    if len(train_df) != len(features):
        raise ValueError("Training rows and behavioural features differ")
    definitions = {
        "target_prior_in_count_24h": _count_bucket(
            features["target_prior_in_count_24h"]
        ),
        "target_prior_in_count_72h": _count_bucket(
            features["target_prior_in_count_72h"]
        ),
        "source_prior_out_count_24h": _count_bucket(
            features["source_prior_out_count_24h"]
        ),
        "source_prior_out_count_72h": _count_bucket(
            features["source_prior_out_count_72h"]
        ),
        "target_amount_to_prior_mean": _ratio_bucket(
            features["target_prior_in_amount_to_prior_mean"]
        ),
        "source_amount_to_prior_mean": _ratio_bucket(
            features["source_prior_out_amount_to_prior_mean"]
        ),
        "target_time_since_last": _dormancy_bucket(
            features["target_prior_in_time_since_last"]
        ),
        "source_time_since_last": _dormancy_bucket(
            features["source_prior_out_time_since_last"]
        ),
    }
    return pd.concat(
        [_summary(train_df, features, name, bucket) for name, bucket in definitions.items()],
        ignore_index=True,
    )


def _plot_rate_panels(
    summary: pd.DataFrame,
    definitions: list[tuple[str, str, tuple[str, ...]]],
    output: str | Path,
    title: str,
    x_label: str,
) -> Path:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(183 * MM_TO_INCH, 125 * MM_TO_INCH),
        squeeze=False,
        sharey=False,
    )
    for ax, (feature, panel_title, buckets) in zip(axes.flat, definitions):
        selected = summary[summary["feature"].eq(feature)]
        x = np.arange(len(buckets))
        for transaction_type, color in zip(GRAPH_TYPES, (BLUE, RED)):
            table = (
                selected[selected["transaction_type"].eq(transaction_type)]
                .set_index("bucket")
                .reindex(buckets)
            )
            rate = table["fraud_rate_pct"].to_numpy(dtype=float)
            ax.plot(
                x,
                rate,
                color=color,
                marker="o",
                markersize=3.0,
                linewidth=1.1,
                label=transaction_type,
            )
            for index, value in enumerate(rate):
                count = table.iloc[index]["transaction_count"]
                fraud = table.iloc[index]["fraud_count"]
                if pd.notna(value) and count:
                    ax.annotate(
                        f"{int(fraud)}/{int(count):,}",
                        (index, value),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=5.8,
                        color=color,
                    )
        ax.set_title(panel_title, fontweight="bold")
        ax.set_xticks(x, buckets, rotation=28)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Fraud rate (%)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.7)
    axes[0, 0].legend(loc="upper right", handlelength=1.8)
    fig.suptitle(title, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "Training-only EDA; labels show fraud count / all transactions; history uses strictly earlier hours.",
        ha="center",
        fontsize=7.5,
    )
    return _save_pdf(fig, output)


def plot_behavioral_eda(
    train_df: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: str | Path = "image/图拓扑特征",
    file_prefix: str = "gfp_behavioral",
    summary_filename: str = "behavioral_history_binned_summary.csv",
) -> dict[str, Path]:
    """Create report-ready PDFs for burst, amount-anomaly and dormancy EDA."""
    summary = summarize_behavioral_eda(train_df, features)
    output_dir = Path(output_dir)
    outputs = {
        "summary": output_dir.parent.parent / "outputs" / "metrics" / summary_filename,
    }
    outputs["summary"].parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outputs["summary"], index=False)

    outputs["burst"] = _plot_rate_panels(
        summary,
        [
            ("target_prior_in_count_24h", "Recipient incoming count: 24h", COUNT_LABELS),
            ("target_prior_in_count_72h", "Recipient incoming count: 72h", COUNT_LABELS),
            ("source_prior_out_count_24h", "Initiator outgoing count: 24h", COUNT_LABELS),
            ("source_prior_out_count_72h", "Initiator outgoing count: 72h", COUNT_LABELS),
        ],
        output_dir / f"{file_prefix}_burst_fraud_rates.pdf",
        "Short-term transaction bursts and fraud rate",
        "Prior transactions in window",
    )
    outputs["amount"] = _plot_rate_panels(
        summary,
        [
            ("target_amount_to_prior_mean", "Recipient current amount / prior mean", RATIO_LABELS),
            ("source_amount_to_prior_mean", "Initiator current amount / prior mean", RATIO_LABELS),
            ("target_time_since_last", "Recipient time since last activity", DORMANCY_LABELS),
            ("source_time_since_last", "Initiator time since last activity", DORMANCY_LABELS),
        ],
        output_dir / f"{file_prefix}_amount_and_dormancy_fraud_rates.pdf",
        "Relative amount and account dormancy patterns",
        "Label-independent history bin",
    )
    return outputs


__all__ = [
    "build_behavioral_history_features",
    "plot_behavioral_eda",
    "summarize_behavioral_eda",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="train_fraud.parquet")
    parser.add_argument("--output-dir", default="image/图拓扑特征")
    args = parser.parse_args()

    train = pd.read_parquet(args.train)
    behavioral_features = build_behavioral_history_features(train)
    generated = plot_behavioral_eda(train, behavioral_features, args.output_dir)
    for name, path in generated.items():
        print(f"{name}: {path}")

    # A matched-period view reduces the obvious coverage confounding in this
    # dataset: hours 635–718 contain both classes, whereas later hours are
    # fraud-only in the supplied training file.
    matched_mask = train["hours_elapsed"].between(635, 718).to_numpy()
    matched = plot_behavioral_eda(
        train.loc[matched_mask].reset_index(drop=True),
        behavioral_features.loc[matched_mask].reset_index(drop=True),
        args.output_dir,
        file_prefix="gfp_behavioral_hours_635_718",
        summary_filename="behavioral_history_binned_summary_hours_635_718.csv",
    )
    for name, path in matched.items():
        print(f"matched_{name}: {path}")
