# Minimal GFP-style features

`graph_features.py` implements five **historical vertex/edge statistics** from
the Graph Feature Preprocessor (GFP) idea: source prior outgoing degree and
distinct-neighbor fan, target prior incoming degree and distinct-neighbor fan,
and prior transactions on the exact directed pair. The edge direction is the
recorded `initiating_customer -> recipient_customer` relationship. For some
transaction types this may not represent the complete economic cash flow.

All counts use transactions with `hours_elapsed` **strictly less than** the
current row's hour. Rows in the same hour receive the same history snapshot.
The implementation does not use `is_fraud`, post-transaction balances, or the
external test file. It keeps input row order and index.

This is a deliberately small **GFP-inspired subset**, not the IBM Snap ML
`GraphFeaturePreprocessor` implementation and not its cycle or scatter-gather
miner. Snap ML is absent from the required Python 3.12 / Apple ARM environment;
the documented Mac package support is Intel. NetworkX is used in a small
reference test; pandas handles the full dataset to avoid a huge Python graph.

The notebook imports `build_gfp_features`, summarizes features against the
training label, and writes PDF figures plus a CSV summary. Those are
descriptive full-training-set comparisons. For model validation, calculate
features from the permitted chronological history in each fold, compare a
tabular baseline with the same model plus these features, and keep any future
external test data isolated until model selection is final.

Sources: the three papers in `paper/`, especially *Graph Feature
Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime
Detection*, and the [Snap ML 1.15 GFP documentation](https://snapml.readthedocs.io/en/v1.15/graph_preprocessor.html).

The synthetic-transaction paper motivates directed transaction multigraphs
and historical pattern features. The network-analytics review motivates
comparing simple manual network features before using GNNs and cautions about
optimistic findings on synthetic data. The GFP paper motivates vertex fan and
degree statistics in a transaction stream. None of the papers establishes that
this challenge's `is_fraud` labels represent AML layering.

Run the notebook from the repository root with the required `torch` Python
kernel. It writes `outputs/metrics/gfp_feature_summary.csv`, a time-matched
summary for hours 635–718, and three PDF figures in
`image/图拓扑特征/`. Run the small reference check with:

```bash
/Users/zhaobo/anaconda3/envs/torch/bin/python code/GFP/test_graph_features.py
```
