# GFP-inspired graph features

These files are a small, pure-Python implementation of selected ideas from
Snap ML's Graph Feature Preprocessor paper. They are not a full GFP replacement.

## Implemented features

- `graph_features.py`: prior-hour incoming/outgoing transaction counts,
  distinct counterparties, and prior count for the same directed pair.
- `gfp_additional_features.py`: prior-hour outgoing/incoming amount sum, mean,
  maximum, and the count of earlier reverse-direction edges (two-edge temporal
  cycles).
- `code/plot/gfp_topology_eda.py`: report-sized label-distribution, fraud-rate,
  and enrichment plots for all twelve features.

All rows in the same `hours_elapsed` bucket see the same history: transactions
within an hour are not ordered arbitrarily. Features do not read `is_fraud` or
post-transaction balances. A zero means no earlier history for the amount
features; use the existing prior-degree features to distinguish that case.

## Example

```python
import pandas as pd
from graph_features import build_gfp_features
from gfp_additional_features import build_additional_gfp_features

transactions = pd.read_parquet("data/raw/train_fraud.parquet")
features = pd.concat(
    [build_gfp_features(transactions),
     build_additional_gfp_features(transactions)],
    axis=1,
)
```

For a chronological validation split, pass the permitted earlier history and
the rows being scored together, then keep features for the scored rows. Never
include later rows in the input. Keep the external test file out of feature
selection and model validation; use it only for final inference.

Snap ML's Python wrapper delegates GFP operations to native `gf_*` functions.
The Windows wheel inspected for this project does not include those functions,
so this code reproduces a limited set of paper-inspired statistics in Python.
It does not implement the native streaming engine, scatter-gather motifs, or
the paper's complete pattern miner.
