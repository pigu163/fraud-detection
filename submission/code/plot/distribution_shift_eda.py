"""Training-only temporal distribution-shift summaries and PDF figures."""

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from plot_style import (
    ACCENT_RED,
    BLACK,
    CATEGORICAL,
    GREY,
    MM_TO_INCH,
    save_pdf_figure,
)


TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TIME_PERIODS = ["Early", "Middle", "Late"]
NUMERIC_SHIFT_COLUMNS = [
    "transaction_amount",
    "initiater_balance_before",
    "initiater_balance_after",
    "recipient_balance_before",
    "recipient_balance_after",
]
NUMERIC_LABELS = {
    "transaction_amount": "Transaction amount",
    "initiater_balance_before": "Initiator balance before",
    "initiater_balance_after": "Initiator balance after",
    "recipient_balance_before": "Recipient balance before",
    "recipient_balance_after": "Recipient balance after",
}


def assign_time_periods(
    df: pd.DataFrame,
    time_column: str = "hours_elapsed",
) -> pd.Series:
    """Assign three equal-width elapsed-hour windows without using test data."""
    if time_column not in df.columns:
        raise KeyError(f"Missing time column: {time_column}")
    hours = pd.to_numeric(df[time_column], errors="raise")
    minimum = int(hours.min())
    maximum = int(hours.max())
    span = maximum - minimum + 1
    period_index = np.minimum(((hours.to_numpy() - minimum) * 3 // span), 2).astype(int)
    return pd.Series(
        pd.Categorical.from_codes(period_index, categories=TIME_PERIODS, ordered=True),
        index=df.index,
        name="time_period",
    )


def _period_metadata(df: pd.DataFrame, periods: pd.Series) -> pd.DataFrame:
    metadata = (
        pd.DataFrame({"time_period": periods, "hours_elapsed": df["hours_elapsed"]})
        .groupby("time_period", observed=True)
        .agg(
            start_hour=("hours_elapsed", "min"),
            end_hour=("hours_elapsed", "max"),
        )
        .reindex(TIME_PERIODS)
    )
    return metadata


def summarize_temporal_shift(
    df: pd.DataFrame,
    numeric_columns: Optional[Iterable[str]] = None,
) -> dict:
    """Summarize temporal composition, label prevalence, JSD and KS distance.

    Numeric KS comparisons use log1p-transformed values and compare the Early
    and Late windows. The full rows in each window are used in the calculations.
    """
    required = {"hours_elapsed", "transaction_type", "is_fraud"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    numeric_columns = list(numeric_columns or NUMERIC_SHIFT_COLUMNS)
    missing_numeric = [column for column in numeric_columns if column not in df.columns]
    if missing_numeric:
        raise KeyError(f"Missing numeric columns: {missing_numeric}")

    periods = assign_time_periods(df)
    metadata = _period_metadata(df, periods)
    work = df.assign(time_period=periods)
    period_summary = (
        work.groupby("time_period", observed=True)
        .agg(
            transaction_count=("is_fraud", "size"),
            fraud_count=("is_fraud", "sum"),
            fraud_rate_pct=("is_fraud", lambda values: values.mean() * 100),
        )
        .reindex(TIME_PERIODS)
        .join(metadata)
    )

    composition = pd.crosstab(
        work["time_period"],
        work["transaction_type"],
        normalize="index",
    ).reindex(index=TIME_PERIODS, columns=TRANSACTION_TYPES, fill_value=0)
    composition_pct = composition.mul(100)
    early_type = composition.loc["Early"].to_numpy(dtype=float)
    late_type = composition.loc["Late"].to_numpy(dtype=float)
    type_js_distance = float(jensenshannon(early_type, late_type))

    numeric_rows = []
    early_mask = periods == "Early"
    late_mask = periods == "Late"
    for column in numeric_columns:
        early_raw = pd.to_numeric(df.loc[early_mask, column], errors="raise").to_numpy()
        late_raw = pd.to_numeric(df.loc[late_mask, column], errors="raise").to_numpy()
        early_log = np.log1p(early_raw)
        late_log = np.log1p(late_raw)
        ks_result = ks_2samp(early_log, late_log, method="asymp")
        early_median = float(np.median(early_raw))
        late_median = float(np.median(late_raw))
        numeric_rows.append(
            {
                "column": column,
                "early_median": early_median,
                "late_median": late_median,
                "ks_statistic": float(ks_result.statistic),
                "ks_p_value": float(ks_result.pvalue),
            }
        )
    numeric_summary = pd.DataFrame(numeric_rows)

    return {
        "period_summary": period_summary,
        "composition_pct": composition_pct,
        "type_js_distance_early_late": type_js_distance,
        "numeric_summary": numeric_summary,
        "period_metadata": metadata,
    }


def plot_temporal_composition_shift(
    df: pd.DataFrame,
    output_dir: str = "image/distribution_shift",
) -> Path:
    """Plot transaction-type composition and fraud prevalence by time window."""
    summary = summarize_temporal_shift(df)
    composition = summary["composition_pct"]
    period_summary = summary["period_summary"]
    x = np.arange(len(TIME_PERIODS))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 * MM_TO_INCH, 92 * MM_TO_INCH),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )

    bottom = np.zeros(len(TIME_PERIODS))
    for index, transaction_type in enumerate(TRANSACTION_TYPES):
        values = composition[transaction_type].to_numpy()
        axes[0].bar(
            x,
            values,
            bottom=bottom,
            width=0.62,
            color=CATEGORICAL[index],
            label=transaction_type,
        )
        bottom += values
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TIME_PERIODS)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Transaction composition (%)")
    axes[0].set_title("Transaction mix", pad=9, fontweight="bold")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        fontsize=7,
        columnspacing=0.8,
    )

    counts = period_summary["transaction_count"].to_numpy()
    fraud_rates = period_summary["fraud_rate_pct"].to_numpy()
    count_bars = axes[1].bar(
        x,
        counts,
        width=0.58,
        color=CATEGORICAL[0],
        alpha=0.88,
        label="Transaction count",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(TIME_PERIODS)
    from matplotlib.ticker import FuncFormatter
    axes[1].set_ylabel("Transactions (millions)")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}"))
    axes[1].set_title("Volume and fraud rate", pad=9, fontweight="bold")
    axes[1].set_ylim(0, counts.max() * 1.35)
    for bar, count in zip(count_bars, counts):
        axes[1].annotate(
            f"{int(count):,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=BLACK,
        )

    fraud_axis = axes[1].twinx()
    fraud_line = fraud_axis.plot(
        x,
        fraud_rates,
        color=ACCENT_RED,
        marker="o",
        markersize=4.5,
        linewidth=1.5,
        label="Fraud rate",
    )[0]
    fraud_axis.set_ylabel("Fraud rate (%)", color=ACCENT_RED)
    fraud_axis.tick_params(axis="y", colors=ACCENT_RED)
    fraud_axis.spines["top"].set_visible(False)
    fraud_axis.set_ylim(0, max(fraud_rates.max() * 1.35, 0.1))
    for position, rate in zip(x, fraud_rates):
        fraud_axis.annotate(
            f"{rate:.3f}%",
            xy=(position, rate),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=ACCENT_RED,
        )
    axes[1].legend(
        [count_bars, fraud_line],
        ["Transaction count", "Fraud rate"],
        loc="upper left",
        bbox_to_anchor=(0, -0.20),
        ncol=2,
    )

    fig.suptitle(
        "Changes across three training-time windows",
        y=1.02,
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(pad=1.2)
    return save_pdf_figure(fig, Path(output_dir) / "temporal_composition_shift")


def _binned_ecdf(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute an ECDF from all observations using common display bins."""
    counts, bin_edges = np.histogram(values, bins=edges)
    cumulative = np.cumsum(counts) / counts.sum()
    return bin_edges[1:], cumulative


def plot_numeric_distribution_shift(
    df: pd.DataFrame,
    output_dir: str = "image/distribution_shift",
    numeric_columns: Optional[Iterable[str]] = None,
) -> Path:
    """Plot log1p ECDFs for numeric fields across three train-time windows."""
    numeric_columns = list(numeric_columns or NUMERIC_SHIFT_COLUMNS)
    summary = summarize_temporal_shift(df, numeric_columns=numeric_columns)
    periods = assign_time_periods(df)
    period_colors = {"Early": CATEGORICAL[0], "Middle": CATEGORICAL[3], "Late": ACCENT_RED}

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(183 * MM_TO_INCH, 140 * MM_TO_INCH),
    )
    axes_flat = axes.ravel()
    for index, column in enumerate(numeric_columns):
        ax = axes_flat[index]
        transformed = np.log1p(pd.to_numeric(df[column], errors="raise").to_numpy())
        finite_values = transformed[np.isfinite(transformed)]
        edges = np.linspace(finite_values.min(), finite_values.max(), 241)
        for period in TIME_PERIODS:
            period_values = transformed[periods.to_numpy() == period]
            x_values, cdf = _binned_ecdf(period_values, edges)
            ax.step(
                x_values,
                cdf,
                where="post",
                color=period_colors[period],
                linewidth=1.4,
                label=period,
            )
        ks_value = float(
            summary["numeric_summary"]
            .set_index("column")
            .loc[column, "ks_statistic"]
        )
        ax.text(
            0.97,
            0.07,
            f"Early / Late\nKS = {ks_value:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color=BLACK,
        )
        ax.set_title(NUMERIC_LABELS.get(column, column).replace(" balance ", " balance\n"),
                     pad=8, fontweight="bold")
        ax.set_xlabel("log1p(value)")
        ax.set_ylabel("Cumulative fraction")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(edges[0], edges[-1])

    axes_flat[-1].axis("off")
    legend_handles = [
        plt.Line2D([0], [0], color=period_colors[period], linewidth=1.6)
        for period in TIME_PERIODS
    ]
    axes_flat[-1].legend(
        legend_handles,
        TIME_PERIODS,
        loc="upper left",
        title="Training-time window",
    )
    metadata = summary["period_metadata"]
    note = "\n".join(
        f"{period}: hours {int(metadata.loc[period, 'start_hour'])}–"
        f"{int(metadata.loc[period, 'end_hour'])}"
        for period in TIME_PERIODS
    )
    axes_flat[-1].text(
        0.02,
        0.42,
        f"{note}\n\nCurves use all observations\nwithin each window.\n\nKS compares Early and Late.",
        transform=axes_flat[-1].transAxes,
        va="top",
        fontsize=8,
        color=BLACK,
        linespacing=1.45,
    )
    fig.suptitle(
        "Numeric distribution shift across training-time windows",
        y=1.01,
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(pad=1.2)
    return save_pdf_figure(fig, Path(output_dir) / "numeric_distribution_shift")
