"""Small, label-free GFP-inspired features for hourly transaction data.

This module implements a subset of the paper's vertex statistics and
temporal-cycle ideas. It is not a replacement for Snap ML's native GFP.
Every feature for hour h uses only transactions from hours strictly less
than h; transactions within the same hour are treated as a batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "hours_elapsed",
    "initiating_customer",
    "recipient_customer",
    "transaction_amount",
)

FEATURE_COLUMNS = (
    "gfp_source_prior_out_amount_sum",
    "gfp_source_prior_out_amount_mean",
    "gfp_source_prior_out_amount_max",
    "gfp_target_prior_in_amount_sum",
    "gfp_target_prior_in_amount_mean",
    "gfp_target_prior_in_amount_max",
    "gfp_prior_reciprocal_cycle_count",
)


def _prior_account_amounts(
    transactions: pd.DataFrame,
    account_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Get per-account amounts from strictly earlier hours."""
    # One row per account-hour ensures every transaction in an hour sees the
    # same snapshot, regardless of input order within that hour.
    hourly = (
        transactions.groupby(
            [account_column, "hours_elapsed"], sort=False, as_index=False
        )
        .agg(
            edge_count=("transaction_amount", "size"),
            amount_sum=("transaction_amount", "sum"),
            amount_max=("transaction_amount", "max"),
        )
        .sort_values([account_column, "hours_elapsed"], kind="stable")
        .reset_index(drop=True)
    )
    groups = hourly.groupby(account_column, sort=False)

    # Subtract the current hour so its transactions never enter their own
    # historical count or amount sum.
    hourly["prior_count"] = groups["edge_count"].cumsum() - hourly["edge_count"]
    hourly["prior_amount_sum"] = (
        groups["amount_sum"].cumsum() - hourly["amount_sum"]
    )

    # A cumulative maximum shifted by one account-hour gives the maximum
    # amount seen strictly before the current hour.
    running_max = groups["amount_max"].cummax()
    hourly["prior_amount_max"] = running_max.groupby(
        hourly[account_column], sort=False
    ).shift(fill_value=0)

    # Mean is zero when there is no earlier account history.
    hourly["prior_amount_mean"] = np.divide(
        hourly["prior_amount_sum"],
        hourly["prior_count"],
        out=np.zeros(len(hourly), dtype=np.float64),
        where=hourly["prior_count"].to_numpy() > 0,
    )

    # Map each original row back to its account-hour summary without changing
    # the input row order or index (the caller may have duplicate index labels).
    query_keys = pd.MultiIndex.from_arrays(
        [transactions[account_column].to_numpy(),
         transactions["hours_elapsed"].to_numpy()],
        names=[account_column, "hours_elapsed"],
    )
    lookup = hourly.set_index([account_column, "hours_elapsed"])
    result = lookup.reindex(query_keys)
    return pd.DataFrame(
        {
            f"{prefix}_prior_amount_sum": result["prior_amount_sum"].to_numpy(),
            f"{prefix}_prior_amount_mean": result["prior_amount_mean"].to_numpy(),
            f"{prefix}_prior_amount_max": result["prior_amount_max"].to_numpy(),
        },
        index=transactions.index,
    )


def _prior_reciprocal_counts(transactions: pd.DataFrame) -> np.ndarray:
    """Count earlier reverse edges, making two-edge temporal cycles."""
    # Reverse each recorded edge: for A -> B, retain historical B -> A edges.
    reverse_edges = pd.DataFrame(
        {
            "source": transactions["recipient_customer"].to_numpy(),
            "target": transactions["initiating_customer"].to_numpy(),
            "hour": transactions["hours_elapsed"].to_numpy(),
        }
    )
    hourly = (
        reverse_edges.groupby(["source", "target", "hour"], sort=False)
        .size()
        .rename("edge_count")
        .reset_index()
        .sort_values(["hour", "source", "target"], kind="stable")
        .reset_index(drop=True)
    )
    hourly["cumulative_count"] = hourly.groupby(
        ["source", "target"], sort=False
    )["edge_count"].cumsum()

    # Query the reverse direction and use only strictly earlier hours. An
    # as-of join is needed because the current pair may have no edge at its
    # current hour, even though earlier reverse edges exist.
    queries = pd.DataFrame(
        {
            "source": transactions["initiating_customer"].to_numpy(),
            "target": transactions["recipient_customer"].to_numpy(),
            "hour": transactions["hours_elapsed"].to_numpy(),
            "_row_position": np.arange(len(transactions)),
        }
    ).sort_values("hour", kind="stable")
    history = hourly[
        ["source", "target", "hour", "cumulative_count"]
    ].sort_values("hour", kind="stable")
    matched = pd.merge_asof(
        queries,
        history,
        on="hour",
        by=["source", "target"],
        direction="backward",
        allow_exact_matches=False,
    )
    counts = np.zeros(len(transactions), dtype=np.int64)
    found = matched["cumulative_count"].notna()
    counts[matched.loc[found, "_row_position"].to_numpy()] = matched.loc[
        found, "cumulative_count"
    ].to_numpy(dtype=np.int64)
    return counts


def build_additional_gfp_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Build prior-hour amount statistics and reciprocal-cycle counts.

    No label or post-transaction balance is read. All input rows are retained,
    and the returned frame has the same row order and index as ``transactions``.
    Pass only transactions available to the scoring process; earlier rows may
    supply history, while same-hour rows never affect one another.
    """
    missing = set(REQUIRED_COLUMNS).difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if transactions.empty:
        raise ValueError("Cannot build graph features from an empty dataset")

    history_columns = list(REQUIRED_COLUMNS)
    if transactions[history_columns].isna().any().any():
        raise ValueError("Graph feature input contains missing values")
    hours = transactions["hours_elapsed"]
    if not pd.api.types.is_integer_dtype(hours) or (hours < 0).any():
        raise ValueError("hours_elapsed must contain non-negative integer hours")
    if not pd.api.types.is_numeric_dtype(transactions["transaction_amount"]):
        raise ValueError("transaction_amount must be numeric")
    if not np.isfinite(transactions["transaction_amount"].to_numpy()).all():
        raise ValueError("transaction_amount must contain only finite values")

    # Use a small internal frame. Labels and after-transaction balances are
    # intentionally excluded from all state and feature calculations.
    history = transactions.loc[:, list(REQUIRED_COLUMNS)].copy()
    source = _prior_account_amounts(
        history,
        "initiating_customer",
        "gfp_source_out",
    ).rename(
        columns={
            "gfp_source_out_prior_amount_sum": FEATURE_COLUMNS[0],
            "gfp_source_out_prior_amount_mean": FEATURE_COLUMNS[1],
            "gfp_source_out_prior_amount_max": FEATURE_COLUMNS[2],
        }
    )
    target = _prior_account_amounts(
        history,
        "recipient_customer",
        "gfp_target_in",
    ).rename(
        columns={
            "gfp_target_in_prior_amount_sum": FEATURE_COLUMNS[3],
            "gfp_target_in_prior_amount_mean": FEATURE_COLUMNS[4],
            "gfp_target_in_prior_amount_max": FEATURE_COLUMNS[5],
        }
    )
    reciprocal = pd.Series(
        _prior_reciprocal_counts(history),
        index=transactions.index,
        name=FEATURE_COLUMNS[6],
    )
    return pd.concat([source, target, reciprocal], axis=1).loc[
        :, list(FEATURE_COLUMNS)
    ]
