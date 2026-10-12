# Graph-feature EDA findings and figure QA

All numbers below were produced from `train_fraud.parquet` by the executed
notebook. The external test dataset was not read for this analysis. Graph
features use only edges from strictly earlier hours and do not use labels.
The figures compare labels descriptively; they are **not** model-validation
results and do not establish causal or production fraud patterns.

## Main finding

For `TRANSFER`, a target with **zero distinct prior initiators** appears in
98 of 107 labelled fraud transactions and 27,142 of 337,462 legitimate
transactions. The resulting full-training fraud rate is 98/27,240 = 0.360%
for this group, versus 107/337,569 = 0.0317% for all transfers. Within the
common hours 635–718, the zero-fan group has 98 frauds among 346 transfers
(28.3%); this high prevalence reflects the unusual late-period label
concentration as well as the feature association.

For `CASH_OUT`, zero prior target fan is not the dominant fraud pattern:
28/250 frauds and 142,930/1,319,946 legitimate transactions fall there.
The difference between transaction types is why the EDA separates them.

Source outgoing fan has a weaker and noisier relationship. Its largest
`CASH_OUT` bin rate is 2/1,587 for the 3–5 bin, so it should not be treated
as strong evidence. No GFP contribution to a trained model has yet been
measured.

## Figure verification

- Three figures were generated from all rows, without sampling, as vector
  PDFs only.
- The PDF files each contain one page and selectable text. The pages were
  visually checked for clipped labels, legible denominators, consistent axes,
  and the distinction between within-label shares and fraud rates.
- Bars show exact fraud/total counts; dashed lines show the base fraud rate
  within each transaction type and displayed time scope. Empty bins are not
  interpreted as zero-risk evidence.
- The plot inputs are the 60-row full-training summary and the separate
  60-row hours-635–718 summary, both saved under `outputs/metrics/`.

## Limitations before modeling

The source and target direction follows the recorded account roles, which
may not equal the economic direction of money for every transaction type.
Hour-level timestamps prevent ordering events within an hour. History-based
features must be rebuilt under the chosen validation protocol; an EDA chart
from the full training file cannot substitute for held-out evaluation.
