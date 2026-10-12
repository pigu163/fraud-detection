"""Leakage-safe, GFP-inspired transaction graph features.

The directed edge is ``initiating_customer -> recipient_customer``.  Each
feature for hour h uses edges from hours strictly smaller than h.  This is a
small vertex-statistics subset of Snap ML's GraphFeaturePreprocessor, not an
implementation of its cycle or scatter-gather pattern miner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("hours_elapsed", "initiating_customer", "recipient_customer")
FEATURE_COLUMNS = (
    "gfp_source_prior_out_degree",
    "gfp_source_prior_out_fan",
    "gfp_target_prior_in_degree",
    "gfp_target_prior_in_fan",
    "gfp_pair_prior_count",
)


def _endpoint_history(
    edges: pd.DataFrame, account: str, counterparty: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return prior edge and distinct-neighbor counts in original row order."""
    ordered = edges.sort_values([account, "hours_elapsed"], kind="stable")
    account_group = ordered.groupby(account, sort=False)
    hour_group = ordered.groupby([account, "hours_elapsed"], sort=False)

    # Subtract all preceding edges in the *same* hour.  Thus no arbitrary
    # ordering of transactions within an hour enters any feature.
    degree = account_group.cumcount() - hour_group.cumcount()

    # The first occurrence of each directed account-counterparty pair adds
    # one distinct neighbor.  The two cumulative sums cancel current-hour
    # additions, including when several new neighbors appear in that hour.
    first_neighbor = (~ordered.duplicated([account, counterparty])).astype("int32")
    fan = (
        first_neighbor.groupby(ordered[account], sort=False).cumsum()
        - first_neighbor.groupby(
            [ordered[account], ordered["hours_elapsed"]], sort=False
        ).cumsum()
    )

    position = ordered["_row_position"].to_numpy()
    degree_out = np.empty(len(edges), dtype=np.int32)
    fan_out = np.empty(len(edges), dtype=np.int32)
    degree_out[position] = degree.to_numpy(dtype=np.int32)
    fan_out[position] = fan.to_numpy(dtype=np.int32)
    return degree_out, fan_out, ordered


def build_gfp_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build five prior-hour graph features without reading labels or test data.

    Rows and index are preserved.  The caller should generate features from
    the permitted chronological history for each modeling split.  For EDA,
    the input is the supplied training file only.
    """
    missing = set(REQUIRED_COLUMNS).difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing graph columns: {sorted(missing)}")
    if transactions.empty:
        raise ValueError("Cannot build graph features from an empty dataset")
    if transactions.loc[:, list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Graph columns contain missing values")
    hours = transactions["hours_elapsed"]
    if not pd.api.types.is_integer_dtype(hours) or (hours < 0).any():
        raise ValueError("hours_elapsed must contain non-negative integer hours")

    edges = transactions.loc[:, list(REQUIRED_COLUMNS)].copy()
    edges["_row_position"] = np.arange(len(edges), dtype=np.int32)

    source_degree, source_fan, source_ordered = _endpoint_history(
        edges, "initiating_customer", "recipient_customer"
    )
    target_degree, target_fan, _ = _endpoint_history(
        edges, "recipient_customer", "initiating_customer"
    )
    pair_group = source_ordered.groupby(
        ["initiating_customer", "recipient_customer"], sort=False
    )
    pair_hour_group = source_ordered.groupby(
        ["initiating_customer", "recipient_customer", "hours_elapsed"],
        sort=False,
    )
    pair_prior = pair_group.cumcount() - pair_hour_group.cumcount()
    pair_count = np.empty(len(edges), dtype=np.int32)
    pair_count[source_ordered["_row_position"].to_numpy()] = pair_prior.to_numpy(
        dtype=np.int32
    )

    result = pd.DataFrame(
        {
            FEATURE_COLUMNS[0]: source_degree,
            FEATURE_COLUMNS[1]: source_fan,
            FEATURE_COLUMNS[2]: target_degree,
            FEATURE_COLUMNS[3]: target_fan,
            FEATURE_COLUMNS[4]: pair_count,
        },
        index=transactions.index,
    )
    return result
