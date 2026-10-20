# Graph-topology EDA findings

These summaries were generated from `train_fraud.parquet` only. There are
3,144,097 transactions and 357 fraud labels (3,143,740 legitimate rows).
The graph features use only transactions from strictly earlier
`hours_elapsed` values; rows sharing an hour are treated as a batch. These are
descriptive training-set results, not held-out model metrics.

## Main distribution differences

- Source prior out-degree, source prior out-fan, and the three source prior
  outgoing amount summaries are non-zero in 98/357 fraud rows (27.45%) and
  381,830/3,143,740 legitimate rows (12.15%). The non-zero rate ratio is
  2.26, so source-side history is the strongest simple positive separation in
  the full training file.
- Target prior in-degree, target prior in-fan, and the three target prior
  incoming amount summaries are non-zero in 231/357 fraud rows (64.71%) and
  2,321,688/3,143,740 legitimate rows (73.85%). Their non-zero rate ratio is
  0.876, so the target-side history is slightly more common in legitimate
  transactions overall.
- The prior same-pair count is non-zero in no fraud row and only 2 legitimate
  rows. The reciprocal-edge count is zero for every row. These two features
  provide no observed positive fraud signal in this dataset and should not be
  presented as useful without a separate validation experiment.

## Transaction-type result

Fraud is observed in `CASH_OUT` and `TRANSFER`. For `TRANSFER`, zero target
prior in-fan contains 98/27,240 fraud transactions (0.3598%), compared with
107/337,569 (0.0317%) across all transfers. This is a strong descriptive
association, but the labels are concentrated late in the observed period: in
the common hours 635-718, the zero-in-fan group contains 98 frauds among 346
transfers (28.3%). The matched-time figure is therefore essential; a model
could otherwise exploit temporal coverage rather than a portable graph signal.

For `CASH_OUT`, zero target prior in-fan contains 28/142,958 frauds (0.0196%),
while the overall CASH_OUT rate is 250/1,320,196 (0.0189%). The target-fan
association is therefore transaction-type specific.

## Files produced

- `gfp_topology_label_distributions.pdf`: within-label distributions for all
  12 implemented features.
- `gfp_topology_fraud_rates.pdf`: fraud rates by feature bin for CASH_OUT and
  TRANSFER, with exact bin counts in the accompanying CSV.
- `gfp_topology_nonzero_enrichment.pdf`: compact non-zero-rate comparison;
  grey rows have no observed non-zero fraud examples.
- `gfp_topology_label_distributions_hours_635_718.pdf`: time-matched
  descriptive distribution check.
- `outputs/metrics/gfp_topology_feature_summary.csv` and
  `gfp_topology_binned_summary.csv`: numerical summaries used by the figures.

## Short-term behavioural EDA

The new behavioural summaries use account-hour snapshots and fixed 24-hour
and 72-hour windows. A window is closed on the left, so the current hour is
never used to describe the current transaction.

- Recipient short-term inactivity is more promising than sender short-term
  burst count in the full training file. For `TRANSFER`, a zero recipient
  incoming count in the previous 24 hours contains 107/112,352 fraud rows
  (0.0952%), versus 107/337,569 (0.0317%) across all transfers. With a 72-hour
  window the corresponding rate is 106/55,759 (0.1901%). For `CASH_OUT`, the
  24-hour zero-history group contains 242/464,513 fraud rows (0.0521%), versus
  250/1,320,196 (0.0189%) overall.
- The apparent recipient-dormancy association weakens after restricting to
  the mixed-class period 635-718. In that period, the `TRANSFER` 24-hour
  zero-history group contains 102/2,868 fraud rows (3.56%), compared with
  102/2,958 (3.45%) for all transfers. This makes recipient inactivity a
  reasonable candidate feature, but not a standalone fraud rule.
- Sender 24-hour and 72-hour outgoing counts do not add much separation in
  the full file: almost all fraud rows are in the zero-count bin, but that bin
  has nearly the same fraud rate as the corresponding transaction type as a
  whole. Sender dormancy above 72 hours shows a higher descriptive rate in
  some small groups, so it should be tested only under chronological
  validation.
- Relative amount features are useful candidates for conditional analysis.
  In the mixed-class period, recipient current amount at least 5 times its
  prior incoming mean contains 66/123 fraud `CASH_OUT` rows (53.7%), while
  recipient no-history `TRANSFER` rows contain 98/346 fraud rows (28.3%).
  These groups are small and overlap with the strong balance/amount pattern,
  so they should be treated as candidate features rather than rules.

Files produced for this section:

- `gfp_behavioral_burst_fraud_rates.pdf`
- `gfp_behavioral_amount_and_dormancy_fraud_rates.pdf`
- `gfp_behavioral_hours_635_718_burst_fraud_rates.pdf`
- `gfp_behavioral_hours_635_718_amount_and_dormancy_fraud_rates.pdf`
- `outputs/metrics/behavioral_history_binned_summary.csv`
- `outputs/metrics/behavioral_history_binned_summary_hours_635_718.csv`

The plotting and feature-construction code is in
`code/plot/behavioral_history_eda.py`. It generates PDF files only.

The plots are EDA evidence for feature selection, not proof of model benefit.
The next modeling step should rebuild these historical features inside each
chronological training split, compare against a non-GFP baseline using the
same threshold and alert-budget protocol, and report average precision,
fraud-class precision/recall/F1, and false-alert counts on an untouched
internal holdout.
