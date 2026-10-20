"""Training-only visual analysis of strictly earlier-hour graph features."""

from __future__ import annotations

# Typography used for the graph-feature EDA figures.
import matplotlib as mpl
mpl.rcParams.update({
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
})

# Color palette shared with the other report figures.
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Vector-friendly export settings.
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text

from graph_features import FEATURE_COLUMNS as BASIC_FEATURE_COLUMNS
from gfp_additional_features import FEATURE_COLUMNS as ADDITIONAL_FEATURE_COLUMNS


MM_TO_INCH = 1 / 25.4
ALL_FEATURE_COLUMNS = tuple(BASIC_FEATURE_COLUMNS) + tuple(ADDITIONAL_FEATURE_COLUMNS)
COUNT_FEATURES = set(BASIC_FEATURE_COLUMNS) | {
    "gfp_prior_reciprocal_cycle_count",
}
COUNT_LABELS = ("0", "1", "2", "3–5", "6–10", "≥11")
COUNT_EDGES = (-0.5, 0.5, 1.5, 2.5, 5.5, 10.5, np.inf)
TRANSACTION_TYPES = ("CASH_OUT", "TRANSFER")

DISPLAY_NAMES = {
    "gfp_source_prior_out_degree": "Source prior out-degree",
    "gfp_source_prior_out_fan": "Source prior out-fan",
    "gfp_target_prior_in_degree": "Target prior in-degree",
    "gfp_target_prior_in_fan": "Target prior in-fan",
    "gfp_pair_prior_count": "Prior same-pair count",
    "gfp_source_prior_out_amount_sum": "Source prior out-amount sum",
    "gfp_source_prior_out_amount_mean": "Source prior out-amount mean",
    "gfp_source_prior_out_amount_max": "Source prior out-amount maximum",
    "gfp_target_prior_in_amount_sum": "Target prior in-amount sum",
    "gfp_target_prior_in_amount_mean": "Target prior in-amount mean",
    "gfp_target_prior_in_amount_max": "Target prior in-amount maximum",
    "gfp_prior_reciprocal_cycle_count": "Prior reciprocal-edge count",
}


def _save_figure(fig: plt.Figure, filename: str | Path) -> Path:
    """Save a report-sized vector PDF."""
    # Keep the smallest text at a readable report size after PDF placement.
    for artist in fig.findobj(match=Text):
        if artist.get_text():
            artist.set_fontsize(max(7, artist.get_fontsize()))
    fig.tight_layout(pad=1.0)
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output.with_suffix(".pdf")


def _check_inputs(train_df: pd.DataFrame, graph_features: pd.DataFrame) -> None:
    required = {"transaction_type", "is_fraud"}
    missing = required.difference(train_df.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    missing_features = set(ALL_FEATURE_COLUMNS).difference(graph_features.columns)
    if missing_features:
        raise ValueError(f"Graph features are missing columns: {sorted(missing_features)}")
    if len(train_df) != len(graph_features) or not train_df.index.equals(graph_features.index):
        raise ValueError("Training rows and graph-feature rows must have the same order and index")
    if not train_df["is_fraud"].isin([0, 1]).all():
        raise ValueError("is_fraud must be binary")


def _bucket_feature(values: pd.Series, feature: str) -> pd.Series:
    """Create label-independent bins for readable group comparisons."""
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"Feature {feature} contains missing or non-finite values")
    if feature in COUNT_FEATURES:
        return pd.cut(values, bins=COUNT_EDGES, labels=COUNT_LABELS)

    # Amount statistics are long-tailed. Quantiles are computed without labels
    # so the binning itself does not use the fraud outcome.
    quantiles = np.unique(np.quantile(values.to_numpy(dtype=float), [0, .50, .80, .90, .95, .99, 1]))
    if len(quantiles) < 3:
        return pd.Series("constant", index=values.index, dtype="object")
    labels = tuple(f"Q{i}" for i in range(1, len(quantiles)))
    return pd.cut(values, bins=quantiles, labels=labels, include_lowest=True, duplicates="drop")


def summarize_topology_eda(
    train_df: pd.DataFrame,
    graph_features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return label summaries and label-independent feature bins.

    This is descriptive EDA on the supplied training file. It does not fit a
    model or use the external test file. The enrichment column is a compact
    screening statistic, not a held-out precision estimate.
    """
    _check_inputs(train_df, graph_features)
    # Keep enough integer range for grouped fraud counts; int8 would overflow
    # once a bin contains more than 127 positive labels.
    labels = train_df["is_fraud"].to_numpy(dtype=np.int64)
    summary_rows = []
    binned_rows = []
    for feature in ALL_FEATURE_COLUMNS:
        values = graph_features[feature].astype(float)
        bucket = _bucket_feature(values, feature)
        fraud_values = values.loc[train_df["is_fraud"].eq(1)]
        legit_values = values.loc[train_df["is_fraud"].eq(0)]
        fraud_nonzero = int((fraud_values > 0).sum())
        legit_nonzero = int((legit_values > 0).sum())
        fraud_n = len(fraud_values)
        legit_n = len(legit_values)
        fraud_rate = fraud_nonzero / fraud_n if fraud_n else np.nan
        legit_rate = legit_nonzero / legit_n if legit_n else np.nan
        summary_rows.append(
            {
                "feature": feature,
                "display_name": DISPLAY_NAMES[feature],
                "fraud_n": fraud_n,
                "legitimate_n": legit_n,
                "fraud_nonzero_pct": 100 * fraud_rate,
                "legitimate_nonzero_pct": 100 * legit_rate,
                "nonzero_rate_ratio": fraud_rate / legit_rate if legit_rate else np.inf,
                "fraud_median": fraud_values.median(),
                "legitimate_median": legit_values.median(),
                "fraud_mean": fraud_values.mean(),
                "legitimate_mean": legit_values.mean(),
            }
        )
        work = pd.DataFrame(
            {
                "feature": feature,
                "display_name": DISPLAY_NAMES[feature],
                "transaction_type": train_df["transaction_type"].to_numpy(),
                "bucket": bucket.astype(object).to_numpy(),
                "is_fraud": labels,
            }
        )
        grouped = (
            work.groupby(["feature", "display_name", "transaction_type", "bucket"], observed=False)
            .agg(transaction_count=("is_fraud", "size"), fraud_count=("is_fraud", "sum"))
            .reset_index()
        )
        grouped["fraud_rate_pct"] = np.where(
            grouped["transaction_count"] > 0,
            100 * grouped["fraud_count"] / grouped["transaction_count"],
            np.nan,
        )
        grouped["label_total"] = grouped["transaction_type"].map(
            work.groupby("transaction_type", observed=False).size()
        )
        grouped["within_type_share_pct"] = np.where(
            grouped["label_total"] > 0,
            100 * grouped["transaction_count"] / grouped["label_total"],
            np.nan,
        )
        binned_rows.append(grouped)

    return {
        "feature_summary": pd.DataFrame(summary_rows),
        "binned_summary": pd.concat(binned_rows, ignore_index=True),
    }


def plot_topology_label_distributions(
    train_df: pd.DataFrame,
    graph_features: pd.DataFrame,
    output_dir: str | Path = "image/graph_topology_features",
    context: str = "All training hours",
    filename: str = "gfp_topology_label_distributions",
) -> Path:
    """Plot within-label topology distributions for every implemented feature."""
    _check_inputs(train_df, graph_features)
    summary = summarize_topology_eda(train_df, graph_features)["binned_summary"]
    fig, axes = plt.subplots(3, 4, figsize=(183 * MM_TO_INCH, 145 * MM_TO_INCH), squeeze=False)
    for index, feature in enumerate(ALL_FEATURE_COLUMNS):
        ax = axes.flat[index]
        selected = summary[summary["feature"].eq(feature)]
        buckets = list(pd.unique(selected["bucket"].dropna()))
        # Preserve the declared order rather than lexical sorting of interval labels.
        if feature in COUNT_FEATURES:
            buckets = list(COUNT_LABELS)
        x = np.arange(len(buckets))
        for label, color, name in [(0, CATEGORICAL[0], "Legitimate"), (1, ACCENT_RED, "Fraud")]:
            table = selected[selected["transaction_type"].isin(train_df["transaction_type"].unique())]
            table = table.groupby("bucket", observed=False)["transaction_count"].sum().reindex(buckets, fill_value=0)
            label_total = int((train_df["is_fraud"] == label).sum())
            # Rebuild label-specific counts from the raw rows so rare fraud is
            # shown as a within-label distribution rather than absolute volume.
            values = graph_features.loc[train_df["is_fraud"].eq(label), feature]
            label_bucket = _bucket_feature(values, feature)
            counts = label_bucket.value_counts(sort=False).reindex(buckets, fill_value=0)
            share = 100 * counts.to_numpy(dtype=float) / label_total if label_total else np.zeros(len(buckets))
            ax.plot(x, share, color=color, marker="o", markersize=3.2, linewidth=1.0, label=name)
        ax.set_title(DISPLAY_NAMES[feature], fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.set_xticks(x, buckets)
        if index // 4 == 2:
            ax.set_xlabel("Prior-history bin")
            ax.tick_params(axis="x", rotation=35)
        else:
            ax.set_xticklabels([])
        if index % 4 == 0:
            ax.set_ylabel("Within-label share (%)")
        else:
            ax.set_ylabel("")
    axes[0, 0].legend(loc="upper right", handlelength=1.8)
    fig.suptitle("Historical graph-feature distributions by label", fontweight="bold")
    fig.text(
        0.5, 0.005,
        f"Training-only descriptive EDA ({context}); bins are defined without using is_fraud.",
        ha="center",
        fontsize=7.5,
    )
    return _save_figure(fig, Path(output_dir) / filename)


def plot_topology_fraud_rates(
    train_df: pd.DataFrame,
    graph_features: pd.DataFrame,
    output_dir: str | Path = "image/graph_topology_features",
    context: str = "All training hours",
    filename: str = "gfp_topology_fraud_rates",
) -> Path:
    """Plot fraud rates by topology bin, separating the two fraud-bearing types."""
    summary = summarize_topology_eda(train_df, graph_features)["binned_summary"]
    fig, axes = plt.subplots(3, 4, figsize=(183 * MM_TO_INCH, 145 * MM_TO_INCH), squeeze=False)
    for index, feature in enumerate(ALL_FEATURE_COLUMNS):
        ax = axes.flat[index]
        selected = summary[summary["feature"].eq(feature)]
        buckets = list(COUNT_LABELS) if feature in COUNT_FEATURES else list(pd.unique(selected["bucket"].dropna()))
        x = np.arange(len(buckets))
        for transaction_type, color, marker in zip(TRANSACTION_TYPES, CATEGORICAL[:2], ("o", "s")):
            table = selected[selected["transaction_type"].eq(transaction_type)].set_index("bucket").reindex(buckets)
            rate = table["fraud_rate_pct"].to_numpy(dtype=float)
            ax.plot(x, rate, color=color, marker=marker, markersize=3.0, linewidth=1.0, label=transaction_type)
        ax.set_title(DISPLAY_NAMES[feature], fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.set_xticks(x, buckets)
        if index // 4 == 2:
            ax.set_xlabel("Prior-history bin")
            ax.tick_params(axis="x", rotation=35)
        else:
            ax.set_xticklabels([])
        if index % 4 == 0:
            ax.set_ylabel("Fraud rate (%)")
        else:
            ax.set_ylabel("")
    axes[0, 0].legend(loc="upper right", handlelength=1.8)
    fig.suptitle("Fraud rate across historical graph-feature bins", fontweight="bold")
    fig.text(
        0.5, 0.005,
        f"Training-only descriptive EDA ({context}); values are fraud/transactions within each type and bin.",
        ha="center",
        fontsize=7.5,
    )
    return _save_figure(fig, Path(output_dir) / filename)


def plot_topology_enrichment(
    feature_summary: pd.DataFrame,
    output_dir: str | Path = "image/graph_topology_features",
    filename: str = "gfp_topology_nonzero_enrichment",
) -> Path:
    """Summarize how much more often a feature is non-zero among fraud rows."""
    table = feature_summary.copy()
    # A small continuity correction keeps a feature with no observed non-zero
    # legitimate value finite; the raw percentages remain in the CSV.
    fraud = table["fraud_nonzero_pct"].to_numpy(float) / 100
    legit = table["legitimate_nonzero_pct"].to_numpy(float) / 100
    fraud_count = fraud * table["fraud_n"].to_numpy(float)
    legit_count = legit * table["legitimate_n"].to_numpy(float)
    fraud_rate_corrected = (fraud_count + 0.5) / (table["fraud_n"].to_numpy(float) + 1)
    legit_rate_corrected = (legit_count + 0.5) / (table["legitimate_n"].to_numpy(float) + 1)
    ratio = fraud_rate_corrected / legit_rate_corrected
    fraud_count_observed = fraud_count > 0
    # Do not let a continuity correction manufacture a positive signal when
    # no fraud row has a non-zero value. Those rows are shown at zero/grey.
    enrichment = np.where(fraud_count_observed, np.log2(ratio), 0.0)
    table["log2_nonzero_enrichment_corrected"] = enrichment
    table["fraud_nonzero_observed"] = fraud_count_observed
    table = table.sort_values("log2_nonzero_enrichment_corrected")
    fig, ax = plt.subplots(figsize=(183 * MM_TO_INCH, 88 * MM_TO_INCH))
    y = np.arange(len(table))
    values = table["log2_nonzero_enrichment_corrected"].to_numpy()
    colors = [
        ACCENT_RED if value > 0 else (CATEGORICAL[0] if observed else GREY)
        for value, observed in zip(values, table["fraud_nonzero_observed"])
    ]
    ax.barh(y, values, color=colors, height=0.62)
    ax.axvline(0, color=BLACK, linewidth=0.7)
    ax.set_yticks(y, table["display_name"])
    ax.set_xlabel("log2 corrected enrichment of non-zero history in fraud")
    ax.set_title("Which historical graph features separate the labels?", fontweight="bold")
    ax.text(
        0.01, -0.18,
        "Positive values indicate a higher non-zero rate among fraud rows; grey rows have no observed non-zero fraud examples.",
        transform=ax.transAxes,
        fontsize=7.5,
    )
    return _save_figure(fig, Path(output_dir) / filename)

