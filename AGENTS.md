# AGENTS.md — Fraud Detection Machine Learning Challenge

## 1. Project Overview

This repository contains a take-home Machine Learning assignment for a
Data Scientist recruitment process.

The objective is to develop a reliable, reproducible, and interpretable
machine learning pipeline for detecting fraudulent financial transactions.

The project must demonstrate:

- Exploratory Data Analysis (EDA)
- Data quality assessment
- Feature engineering
- Appropriate model selection
- Robust model validation
- Fraud detection performance evaluation
- Business-oriented interpretation
- Reproducible Python implementation
- Professional PDF reporting

The final solution must balance predictive performance, computational
efficiency, interpretability, and practical applicability.

All implementations must use Python.

---

## 2. Python Environment

The project MUST use the following Python interpreter:

`D:\miniconda3\envs\nasdaq\python.exe`

Always use this interpreter when executing Python scripts.

Example:

```powershell
& "D:\miniconda3\envs\nasdaq\python.exe" src/train.py
```

Do not create another virtual environment unless explicitly requested.

Verified compatibility in the Windows `nasdaq` environment:

- Python 3.9.25, `numpy==1.26.4`, and `snapml==1.15.6`.
- Keep NumPy below 2 while using Snap ML 1.15.6; its native extension is incompatible with NumPy 2.
- Snap ML CPU model training and prediction have been verified. Snap ML 1.15.6's `GraphFeaturePreprocessor` is not supported on Windows, even though its Python wrapper is present.

Before installing additional dependencies:

1. Check whether the package is already installed.
2. Prefer packages available in the existing environment.
3. Avoid unnecessary dependencies.
4. Record all required dependencies for reproducibility.

Recommended libraries:

- pandas
- numpy
- polars
- pyarrow
- scikit-learn
- matplotlib
- seaborn
- lightgbm
- xgboost
- catboost
- scipy
- joblib

Additional libraries may be used when justified.

---

## 3. Input Data

The assignment provides three files.

### 3.1 Training Dataset

File:

`train_fraud.parquet`

Source:

https://www.dropbox.com/scl/fi/u1ubkupsi5x2n3xrurwyc/train_fraud.parquet?rlkey=mpcy23ky7c90dg2sfh4uufaoq&dl=0

Description:

Contains financial transaction records and the target variable:

`is_fraud`

This is the ONLY dataset permitted for model development and training.

### 3.2 External Test Dataset

File:

`test_fraud_external.parquet`

Source:

https://www.dropbox.com/scl/fi/qzwi9xk8oh74bnn4olly8/test_fraud_external.parquet?rlkey=174him0vh0njz8qgittav01ds&dl=0

Description:

Contains external transaction records requiring fraud predictions.

IMPORTANT:

- Do not use this dataset for model training.
- Do not use this dataset for hyperparameter optimization.
- Do not use this dataset for feature selection.
- Do not use this dataset for threshold optimization.
- Do not use this dataset to select or compare models.
- Do not use its distributional statistics to fit preprocessing.
- Do not use its labels, if present, for any modeling decisions.

The external test dataset must remain isolated from model development.

It may only be used for final inference and output generation.

### 3.3 Column Descriptions

File:

`columns.txt`

Source:

https://www.dropbox.com/scl/fi/o35rh6b7tb1wasm4z3nw3/columns.txt?rlkey=ezm4ot1hs1vhvhnsi4waeu6r9&dl=0

Read this file before performing feature engineering.

Do not assume column meanings without checking the documentation.

---

## 4. Mandatory Deliverables

The final submission MUST contain exactly the three categories of
deliverables requested by the recruiter.

### Deliverable 1 — Python Source Code

Provide the complete Python implementation, including:

- Data loading
- EDA
- Data preprocessing
- Feature engineering
- Model training
- Model validation
- Hyperparameter optimization, if appropriate
- Evaluation
- Visualization generation
- External test prediction
- Output generation

The code must be:

- Readable
- Modular
- Reproducible
- Appropriately commented
- Free from hardcoded machine-specific data paths
- Executable using the specified Python interpreter

### Deliverable 2 — PDF Report

Generate a professional PDF report describing:

1. Problem definition
2. Dataset overview
3. Exploratory Data Analysis
4. Data quality observations
5. Feature engineering
6. Modeling methodology
7. Validation strategy
8. Evaluation metrics
9. Visualizations
10. Interpretation of results
11. Challenges and limitations
12. Potential improvements
13. Business implications

The report must be understandable to both technical and non-technical
audiences.

All reported numerical results must come from actual experiments.

Never fabricate metrics, findings, or experimental results.

### Deliverable 3 — Predicted External Test Dataset

Generate:

`test_fraud_external.parquet`

The submitted file must contain all original external test columns
plus one additional column:

`is_fraud_predicted`

Requirements:

- Preserve all original rows.
- Preserve original row order.
- Preserve original columns.
- Preserve original column values.
- Append the prediction column.
- Do not introduce missing prediction values.
- Ensure the prediction column contains binary labels (0 or 1).
- Preserve Parquet format.

The final file must be generated using the selected trained model.

Do not modify or overwrite the original downloaded input file.
Save the submission version in a separate output directory.

---

## 5. Suggested Project Structure

Use the following structure unless there is a strong reason to change it.

```text
fraud-detection-challenge/
│
├── AGENTS.md
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── train_fraud.parquet
│   │   ├── test_fraud_external.parquet
│   │   └── columns.txt
│   │
│   └── processed/
│
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── models/
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
│
├── reports/
│   └── fraud_detection_report.pdf
│
└── submission/
    ├── code/
    ├── fraud_detection_report.pdf
    └── test_fraud_external.parquet
```

Avoid unnecessary files, excessive abstraction, and overengineering.

---

## 6. Exploratory Data Analysis

Perform EDA before selecting the final modeling strategy.

### 6.1 Dataset Overview

Analyze:

- Number of observations
- Number of features
- Feature data types
- Target distribution
- Fraud rate
- Memory consumption
- Missing values
- Duplicate records
- Constant features
- High-cardinality categorical features

### 6.2 Target Distribution

Analyze the distribution of:

`is_fraud`

Determine whether the dataset is imbalanced.

Visualize:

- Fraud vs. non-fraud counts
- Fraud rate

Explain the implications for model evaluation.

### 6.3 Numerical Features

Analyze:

- Mean
- Median
- Standard deviation
- Minimum and maximum
- Quantiles
- Missing-value proportions
- Outliers
- Class-conditional distributions

Generate informative plots where appropriate.

### 6.4 Categorical Features

Analyze:

- Category frequencies
- Cardinality
- Missing categories
- Fraud rates by category
- Rare categories

Avoid producing excessively large or unreadable plots.

### 6.5 Temporal Analysis

If timestamps are available, investigate:

- Transaction frequency over time
- Fraud rate over time
- Hour-of-day patterns
- Day-of-week patterns
- Temporal distribution shifts

Use only information available at the time of prediction.

### 6.6 Entity-Level Analysis

If account, customer, merchant, device, or other entity identifiers
are available, investigate:

- Transaction frequency
- Transaction amount patterns
- Entity activity
- Repeated behavior
- Suspicious concentration of transactions

Do not assume entity identifiers exist.

### 6.7 EDA Visualization Requirements

Generate relevant visualizations such as:

- Class distribution
- Missing-value distribution
- Numerical feature distributions
- Fraud vs. non-fraud comparisons
- Transaction amount distributions
- Temporal patterns
- Category-level fraud rates
- Feature correlations, when appropriate

All visualizations must:

- Have descriptive titles
- Have clearly labeled axes
- Use readable fonts
- Be suitable for a professional report
- Avoid misleading scales
- Be saved to `outputs/figures/`

Do not generate plots merely to increase their quantity.

Prioritize informative findings.

---

## 7. Data Leakage Prevention

Data leakage prevention is a CRITICAL requirement.

### 7.1 General Rules

Never use target information directly or indirectly in features.

Check for:

- Target leakage
- Temporal leakage
- Entity leakage
- Duplicate transaction leakage
- Train-validation contamination
- Preprocessing leakage
- Aggregation leakage

### 7.2 Train-Validation Split

Create an internal validation strategy using only
`train_fraud.parquet`.

Select the splitting strategy based on the data:

- Stratified random split when observations can reasonably be
  treated as independent.
- Temporal split when chronological prediction is relevant.
- Group-aware split when shared entities could cause leakage.

Explain and justify the selected strategy.

Do not assume random splitting is always appropriate.

### 7.3 Preprocessing

Fit preprocessing transformations using training data only.

Examples:

- Imputation
- Scaling
- Encoding
- Feature selection
- Dimensionality reduction

Apply fitted transformations to validation data.

Prefer scikit-learn Pipelines or equivalent reproducible mechanisms.

### 7.4 Aggregation Features

Features involving historical transaction statistics must respect
information availability.

For example:

- Historical transaction count
- Historical transaction amount
- Historical average transaction amount
- Historical transaction frequency

When temporal ordering matters, compute these features using
past transactions only.

Never use future transactions to construct historical features.

### 7.5 External Test Isolation

The external test dataset must not influence any modeling decisions.

Final predictions should be generated only after the complete
modeling and validation procedure has been finalized.

---

## 8. Feature Engineering

Feature engineering should be driven by EDA findings and
available dataset columns.

Potential feature categories include:

### 8.1 Transaction Features

If relevant fields exist:

- Transaction amount transformations
- Log-transformed amount
- Amount deviation from historical behavior
- Transaction type
- Transaction channel
- Transaction frequency

### 8.2 Temporal Features

If timestamps exist:

- Hour of day
- Day of week
- Weekend indicator
- Time since previous transaction
- Recent transaction count
- Historical transaction velocity

For this dataset, `hours_elapsed` is an elapsed-hour index, not a
calendar timestamp. Do not interpret `hours_elapsed % 24` as a known
clock hour or infer weekdays/weekends without a documented start time.
Transactions sharing an hour have no known within-hour order.

### 8.3 Behavioral Features

If entity identifiers exist:

- Historical transaction count
- Historical average transaction amount
- Historical maximum transaction amount
- Historical transaction frequency
- Time since last activity
- Deviations from historical behavior

### 8.4 Categorical Features

Consider:

- Native categorical support
- One-hot encoding
- Frequency encoding
- Target encoding with proper leakage prevention

Avoid high-dimensional encodings when they provide limited value.

### 8.5 Graph-Based Features

Graph features are OPTIONAL.

Only introduce graph-based features when the dataset contains
meaningful relational information.

Potential examples:

- In-degree and out-degree
- Transaction network connectivity
- Fan-in and fan-out patterns
- Reciprocal transactions
- Historical interaction counts
- Local structural patterns

Graph methods must not be added solely because they are complex.

They should be evaluated against simpler alternatives.

Any graph-based feature must respect validation boundaries
and temporal information availability.

### 8.6 Feature Engineering Principles

- Prefer informative features over excessive features.
- Avoid unnecessary complexity.
- Validate the contribution of engineered features.
- Ensure feature transformations work on unseen data.
- Keep training and inference feature generation consistent.

### 8.7 Concrete Feature Metrics to Test on This Dataset

These are candidate **input features**, not model-performance metrics.
Add them in groups and keep a feature only if internal validation shows
useful incremental performance. Use the spellings in `columns.txt` and
the actual Parquet schema; do not invent unavailable fields.

**Transaction and pre-transaction balance:**

- `transaction_type`, `transaction_amount`, and
  `log1p(transaction_amount)`.
- `initiater_balance_before` and its zero-balance indicator.
- `transaction_amount / initiater_balance_before` when the denominator
  is positive; give a zero denominator its own indicator/category and
  never create infinity or silently discard those rows.
- An indicator for positive starting balance and absolute difference
  between transaction amount and that balance at most $0.01.
- Indicators for amount exceeding the starting balance and for other
  *pre-specified* amount-to-balance ranges, if validated.
- `recipient_balance_before` and its zero-balance indicator, if this
  field is available at the actual prediction time.
- Do not use `initiater_balance_after` or `recipient_balance_after` in
  a pre-approval model. They are only suitable for a separately named
  post-transaction or hypothetical-completed-balance experiment.

**Strictly earlier-hour account and graph history:**

- Initiator's prior outgoing transaction count and distinct-recipient
  count (out-degree/out-fan); recipient's prior incoming transaction
  count and distinct-initiator count (in-degree/in-fan).
- Prior count for the same initiator-recipient pair.
- Earlier-hour incoming/outgoing amount sums, means or maxima; amount
  of current transaction relative to that account's earlier behavior.
- Time since the initiator's and recipient's last recorded activity.
- Optional rolling counts and amount sums over pre-specified elapsed-
  hour windows, such as the previous 24 or 72 hours. Exclude the
  current hour unless its true transaction order is known.
- Optional prior incoming/outgoing amount ratio and indicators for a
  new counterparty. Define behavior for accounts with no history.

The existing `code/GFP/graph_features.py` implements only a small,
strict-prior-hour vertex-statistics subset. Do not call it the full GFP
pattern miner. Full fan-in/out motifs, two-hop flows, scatter-gather,
cycles, or community features are optional follow-ups. Compute them
only from history available before scoring, and retain them only if
they beat simple features under the same validation protocol.

Training-set EDA found 346 positive-balance full-amount rule matches,
all labelled fraud. That is **not** held-out precision. The graph EDA
suggests recipient prior in-fan may help for `TRANSFER`, but no model
gain has been demonstrated. In particular, do not assume this fraud
label represents an AML laundering stage or an account-level case.

### 8.8 Prioritized Tabular Feature Experiments for the Next Modeling Session

The priorities below are hypotheses supported by training-set EDA, not
measured model improvements. Start with the strongest simple candidates
and test additional groups under the same validation protocol.

| Priority | Candidate group | Evidence and expected role |
| --- | --- | --- |
| 1 | Amount relative to initiator's starting balance, plus full-balance indicator | Strongest observed class separation; make the relationship directly available to a tabular model. |
| 2 | Transaction type and interactions with balance features | Labelled fraud occurs in `CASH_OUT` and `TRANSFER`; determine whether balance relationships differ by type. |
| 3 | Recipient's strictly earlier-hour incoming history | Prior distinct-initiator count is a promising supplementary signal for `TRANSFER`, including some fraud outside the full-balance rule. |
| 4 | Initiator history, amount history, recency, and simple pair history | Test whether behavioral context adds information beyond transaction and balance features. |
| 5 | More complex historical graph patterns | Optional; require an improvement over simple account and pair statistics before retaining them. |

**Priority 1: amount versus starting balance.** Use these proposed names
consistently in modeling code and the saved feature schema. Let `A` be
`transaction_amount` and `B` be `initiater_balance_before`:

- `amount_to_initiater_balance_ratio = A / B` for `B > 0`. For `B == 0`,
  retain the row, mark the ratio as undefined, and provide
  `initiater_balance_before_is_zero`. Use native missing-value support
  or an imputer fitted on the training partition, as appropriate.
  Inspect negative balances explicitly if present; do not silently
  treat them as positive-balance ratios.
- `is_full_balance_amount_match = (B > 0) & (abs(A - B) <= 0.01)`. This
  is a candidate feature for all transaction types; apply type-specific
  interactions separately. Preserve the existing EDA
  definition in `code/plot/full_balance_eda.py`.
- `amount_minus_initiater_balance = A - B` and optionally
  `abs_amount_minus_initiater_balance = abs(A - B)` are alternative
  ways to expose proximity to the starting balance. Test them as
  additions to the ratio group rather than assuming every redundant
  representation is useful.
- `amount_exceeds_initiater_balance = (A > B)` provides a simple
  overspend indicator. Keep it separate from the one-cent full-balance
  tolerance; near-equal amounts can satisfy both definitions.
- `transaction_amount_is_zero` and, optionally,
  `amount_and_initiater_balance_are_zero` distinguish zero-amount
  transactions from positive-balance withdrawals. These are hypotheses
  to test, not automatic fraud rules.
- Ratio bins already used for EDA may be tested as a separate
  categorical representation. Prefer the continuous ratio and the
  full-balance indicator initially; do not tune many bin edges using
  holdout labels.

Existing full-training-set EDA found 346 full-balance rule matches,
all labelled fraud, out of 357 labelled fraud transactions. This makes
the ratio/rule group the leading candidate, but does not establish
future precision or generalization. The 11 fraud rows outside the rule
comprise three zero-amount/zero-starting-balance `CASH_OUT` rows and
eight `TRANSFER` rows. Inspect these cases during development without
building row-specific rules. Document the full-dataset discovery
exposure as required in Section 10.

**Priority 2: type and interactions.** Include `transaction_type` with
appropriate categorical handling. Candidate explicit interactions are
`is_transfer_and_full_balance`, `is_cash_out_and_full_balance`, and
type-specific amount-to-balance behavior. Tree models can learn these
interactions directly; test explicit versions only for incremental
benefit. For a linear baseline, explicit interactions may be useful.
Do not label every `CASH_OUT` or `TRANSFER` transaction as fraud, and do
not assume other transaction types can never be fraudulent in future
data. Keep recipient starting balance and its zero indicator as
lower-priority transaction features when available before scoring.

**Priority 3: recipient incoming history.** First reuse the implemented
features `gfp_target_prior_in_degree` (prior incoming transaction count)
and `gfp_target_prior_in_fan` (prior distinct initiator count). Candidate
indicators include `recipient_has_no_prior_incoming_transactions` and
`recipient_has_no_prior_incoming_counterparties`. These describe absence
of history in the supplied records, not account age or a newly opened
bank account. An optional `TRANSFER` interaction with zero prior in-fan
can be tested. The graph EDA suggests separation in this subgroup, but
the number of fraud observations is small and the subgroup includes
legitimate transactions. Do not interpret it as a deterministic rule.

**Priority 4: behavioral context.** Start with the existing
`gfp_source_prior_out_degree`, `gfp_source_prior_out_fan`, and
`gfp_pair_prior_count`. Add earlier-hour amount sums/means, current
amount relative to prior mean, and time since last activity only after
this simple group has been evaluated. For historical ratios, distinguish
no history from a history with mean amount zero. Fit any global fallback
on training data only. Pair count has little variation in the observed
EDA, so check whether it adds useful information before retaining it.

**Priority 5: advanced graph candidates.** Reciprocal prior transfers,
two-hop flows, fan-in/out motifs, and cycles are follow-up hypotheses.
Use strictly permitted historical edges, document the time window and
pattern definition, and measure computation and inference cost. Current
GFP-inspired code does not implement these motifs. A graph association
with `is_fraud` does not by itself establish money laundering.

**Time-feature caution.** Fraud was observed only in hours 635–743,
while legitimate records end at hour 718. `hours_elapsed` can therefore
look highly predictive by exploiting this dataset's coverage. Use it
for ordering, historical feature construction, and temporal validation.
If included as a model input, compare results with and without it and
evaluate the mixed-class period separately. Do not treat performance in
an all-fraud late period as proof of practical discrimination, or derive
actual hour of day / weekday without a known calendar origin.

**Concrete feature-group comparison to run tomorrow:**

1. Evaluate the full-balance rule separately as a rule baseline; also
   train a simple learned model on permitted raw transaction features.
2. Add the ratio, full-balance indicator, zero-balance indicator, and
   selected balance differences to the learned baseline.
3. Test transaction-type interactions and lower-priority transaction
   indicators without changing the evaluation partition.
4. Add simple account history, reporting recipient incoming-history
   contribution separately where feasible.
5. Add remaining GFP-inspired features; recognize overlap with account
   history. Do not present the same count as an independent new feature.
6. Try advanced graph patterns only if time permits and the preceding
   comparisons justify the cost.

Use the same estimator configuration for the initial feature comparison
so that changed hyperparameters do not obscure feature contributions.
Save feature lists and compare development AP, fraud precision/recall/F1,
and Precision@K / Recall@K at a fixed alert budget. Use the threshold
selection and final holdout procedure in Section 10. Examine whether
new features recover fraud missed by the rule and how many false alerts
they introduce. Inspect feature importance only as supporting evidence;
retain features based on validation performance, stability, and cost.
All preprocessing must be fitted within the training partition, and
training and inference must use identical feature definitions. This
plan does not authorize external test data for feature selection.

---

## 9. Model Development

### 9.1 Baseline Models

Establish a reproducible baseline before experimenting with
complex models.

Suggested baseline:

- Logistic Regression
- Simple tree-based classifier

The baseline provides a reference for performance improvements.

### 9.2 Candidate Models

Consider models appropriate for tabular fraud detection:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

The final model should be selected based on validation results,
not assumptions.

Deep learning is optional and should only be introduced
when justified by the data and experimental evidence.

### 9.3 Class Imbalance

Fraud datasets may be highly imbalanced.

Consider:

- Class weighting
- Scale_pos_weight
- Decision threshold optimization
- Appropriate resampling techniques

Any resampling must be performed on training folds only.

Never apply SMOTE or other resampling methods to validation
or external test data.

### 9.4 Hyperparameter Optimization

Use reasonable hyperparameter search methods when beneficial.

Examples:

- RandomizedSearchCV
- Optuna
- Manual structured experiments

Avoid excessive computational expense.

Use a consistent evaluation metric for model selection.

Set random seeds where supported.

### 9.5 Model Selection

Prefer models that demonstrate:

- Strong fraud detection performance
- Good precision-recall trade-offs
- Stable validation results
- Reasonable computational cost
- Interpretability

Do not select models solely based on accuracy.

---

## 10. Evaluation Strategy

The unit of prediction is a **transaction** and the positive class is
`is_fraud == 1`. Do not silently change this to account-level AML case
detection. The three papers in `paper/` motivate minority-class F1,
precision/recall, PR curves, alert-budget evaluation, and graph-feature
ablations; their reported model scores are not results on this dataset.

### 10.1 Evaluation Contract to Fix Before Training

Write down and save the following before comparing candidate models:

1. The prediction time (pre-approval unless explicitly changed), usable
   fields, target definition, and whether earlier-hour unlabeled
   transactions can update historical features during inference.
2. The development, threshold-selection, and final internal evaluation
   partitions, using `train_fraud.parquet` only. Record hour ranges,
   row counts, fraud counts, fraud rates, and any shared entities.
3. The **primary model-selection metric: Average Precision (AP)**,
   calculated from fraud scores, with a stated operational constraint
   or alert-budget metric when investigation capacity is known.
4. The rule for selecting one operating threshold using development
   data only. Do not choose a threshold after viewing final holdout
   outcomes. If the same validation labels are used repeatedly to
   choose models and thresholds, acknowledge the optimistic bias.
5. The candidate feature groups and baseline models. Compare graph
   features by ablation under the same split and alert budget.

There are 357 positives among 3,144,097 training rows (about
0.0114%). These are full-training-set EDA numbers, **not** the
prevalence of any future validation partition. Recompute every metric
denominator on the exact evaluation partition. A model predicting only
the negative class achieves about 99.9886% accuracy on the full
training set while detecting no fraud; therefore accuracy cannot
select a model. No metric from a paper may be copied as this project's
experimental result.

### 10.2 Confusion Matrix and Threshold-Dependent Metrics

For a *locked* score threshold, define:

- `TP`: actual fraud predicted fraud.
- `FP`: actual legitimate transaction predicted fraud.
- `FN`: actual fraud predicted legitimate.
- `TN`: actual legitimate transaction predicted legitimate.

Report all four **counts** and the number of evaluated rows. Use
`is_fraud == 1` as the positive class everywhere. Derive and report:

- **Precision / positive predictive value** = `TP / (TP + FP)`.
  It answers: of the alerts, how many were actual fraud?
- **Recall / true-positive rate** = `TP / (TP + FN)`.
  It answers: of the actual fraud cases, how many were found?
- **F1 for the fraud class** = `2 * precision * recall /
  (precision + recall)`, or equivalently `2TP / (2TP + FP + FN)`.
  State the positive class explicitly; do not substitute micro-F1 or
  a support-weighted average for fraud-class F1.
- **False-positive rate** = `FP / (FP + TN)`, useful as a diagnostic
  but potentially deceptively small with millions of legitimate rows.
- **Alert count** = `TP + FP`; **alert rate** = `(TP + FP) / N`;
  **false alerts per 100,000 transactions** = `100000 * FP / N`.
  Always show the absolute FP count beside any FPR.
- **False negatives** = `FN`, because missed fraud matters separately
  from alert quality.

The script must handle zero denominators explicitly. If no alerts are
issued, set precision and F1 to zero for reproducible computation and
label precision as undefined/no alerts in the report. If an evaluation
partition has no fraud, recall, F1, AP and ROC-AUC cannot support a
normal model comparison; do not replace them with a reassuring zero or
one. State `not defined` and fix the validation design if possible.

Report fractions or percentages consistently. For example, `0.0114%`
is a prevalence, whereas `0.0114` as a fraction is 100 times larger.
When comparing two percentages, use **percentage points** for their
absolute difference. Always include the numerator and denominator
behind a striking rate, especially a displayed 100% precision from
only a few alerts.

### 10.3 Threshold-Independent Ranking Metrics

**Primary: Average Precision (AP).** Use
`sklearn.metrics.average_precision_score(y_true, fraud_score)` on an
internal evaluation partition with both classes. AP summarizes the
precision-recall ranking over thresholds. In code and the report call
it `Average Precision (AP)`; if the phrase `PR-AUC` is used, define
that it refers to this exact implementation. Do not silently mix AP
with trapezoidal area under an interpolated PR curve, which can give
a different value. Draw the PR curve and mark the partition's fraud
prevalence as the no-skill precision reference. Compare AP with that
*same-partition* prevalence, not the full-training prevalence.

**Secondary: ROC-AUC.** Report it when the evaluation partition
contains both classes. It summarizes positive-versus-negative
ranking, but can look strong despite many false alerts under extreme
imbalance. Include the ROC curve for context, never use ROC-AUC alone
to select the model. Accuracy is optional context only.

Evaluate AP and ROC-AUC on continuous fraud scores/probabilities,
not on hard 0/1 predictions. Record whether scores are calibrated
probabilities or only rankings. Never calculate these metrics on the
external test file because its ground-truth labels are unavailable.

### 10.4 Alert-Budget Metrics for an Investigation Team

If the team can inspect at most `K` alerts in a defined interval,
rank evaluation transactions by fraud score and compute:

- **Precision@K** = fraud cases among the top `K`, divided by `K`.
- **Recall@K** = fraud cases among the top `K`, divided by all actual
  fraud cases in that same evaluation partition.
- **Fraud found at K** and **false alerts at K** as absolute counts.
- Optionally report alerts or false alerts per 100,000 transactions
  so comparisons are meaningful across partitions of different size.

Define the interval, `K`, and tie-breaking rule before inspecting
results. Do not copy a paper's arbitrary top-0.1%, top-1%, or top-10%
cutoff as if it were the bank's real capacity. If no real capacity is
provided, show a *predeclared small set* of illustrative budgets,
clearly marked as scenarios rather than business requirements. A
score threshold and a top-K policy are different operating rules;
state which one produces the submitted binary labels. The network
analytics paper in `paper/` uses top-score fractions to illustrate
limited investigation resources; adapt the idea to this task's much
lower observed fraud prevalence.

### 10.5 Threshold Choice and Its Evaluation

Do not assume `0.5` is appropriate. Select a threshold only from
internal development/calibration labels, using one declared objective:

- Maximize fraud-class F1 **if** that is the chosen objective and no
  operational constraint is available.
- Meet a minimum precision or recall, if such a target is provided.
- Stay within an alert budget and maximize recall or fraud found.
- Minimize expected cost only when credible FP/FN or investigation
  costs are supplied. Do not invent costs to obtain a preferred model.

For every candidate threshold, compute TP, FP, FN, TN, precision,
recall, F1, alert count and alert rate. Plot precision, recall, F1 and
alert count against threshold when useful. Save the selected numeric
threshold and exact selection rule. Apply that frozen threshold to an
untouched internal holdout for the final estimate, and later to the
external file for inference. If sample size prevents a separate
threshold-calibration and final-holdout partition, use a documented
out-of-fold procedure or explicitly label same-set threshold metrics
as exploratory/optimistic. Never optimize on the external file.

### 10.6 Validation Protocol and Uncertainty

Because fraud appears only late in the observed period, assess the
feasibility of chronology-based splits by **counting positive labels
on both sides** before training. Do not train on a zero-positive early
window. Transactions after hour 718 have no legitimate rows in the
observed training data; evaluate that segment separately rather than
letting it dominate the primary estimate of alert precision. A
stratified random split can be a secondary sensitivity analysis, but
must not be presented as equivalent to a forward-time deployment test.

For repeated splits or folds, report the metric for each feasible
split plus a summary; do not hide large variation behind one mean.
Always report positive counts and alert counts beside precision,
recall and AP. Confidence intervals may be added when defensible;
respect temporal/entity dependence in any resampling. With only 357
labelled fraud cases in the entire training file, report uncertainty
and avoid claiming that a small numeric gain is proven robust.

All models, feature groups and baselines must use the **same**
evaluation rows, label definition, score orientation and threshold
selection protocol. Preprocessing and resampling are fit only on the
training portion. Historical features must be reproducible for a
future transaction using only permitted earlier-hour, label-free
history; specify whether online history updates with earlier
validation transactions are simulated.

The current notebook already examined labels across the **entire**
training file and identified the full-balance rule. A holdout created
now can remain untouched for fitting, tuning and threshold selection,
but it is not fully blind to that earlier feature discovery. Disclose
this exploratory selection when reporting its rule or model results;
do not claim an unbiased, independently discovered 100% rule precision
from a later internal split.

### 10.7 Model and Feature-Group Comparison

At a minimum compare:

1. A no-skill reference: fraud prevalence for the PR/AP reference and
   the all-negative classifier for the accuracy/recall warning.
2. The existing `is_flagged_fraud` flag and the positive-balance
   full-amount rule as transparent *reference rules*, evaluated on
   internal holdout rows, with the prior full-data EDA exposure
   disclosed. Their full-training EDA rates are not holdout model
   scores, and an internal holdout cannot erase that prior exposure.
3. A simple learned baseline using transaction and pre-transaction
   balance features.
4. The same learner plus earlier-hour account-history features.
5. The same learner plus candidate graph features. This ablation is
   required before claiming graph methods improve the task.

Use a comparison table with model/feature group, split and positive
count, AP, ROC-AUC (if defined), threshold or `K`, TP, FP, FN, TN,
precision, recall, fraud-class F1, alert count, and false alerts per
100,000 transactions. Include training/inference time if a graph
feature offers only a small gain at substantial cost. Report model
hyperparameters and seeds. Pick the final model mainly by AP and
the declared alert-budget objective, subject to stability and
practicality, not by accuracy or one impressive subgroup rate.

### 10.8 Segment, Error and Calibration Checks

For the selected model, inspect results separately for:

- `CASH_OUT` and `TRANSFER`, while stating the transaction and fraud
  count in each segment. Other types contain no labelled fraud in
  this training file, so a zero observed fraud rate is not proof of
  safety.
- Earlier versus later fraud-bearing hour windows, and the special
  after-hour-718 segment described above.
- Rule-match versus rule-miss transactions, zero versus positive
  starting balance, and accounts with versus without earlier history.
- If graph features are retained, their incremental ability to find
  the fraud cases missed by the simple balance rule **at the same
  number of alerts**, not just their univariate fraud-rate contrast.

Show the false-positive and false-negative counts for each meaningful
segment; avoid interpreting unstable rates from tiny denominators.
Inspect representative errors without exposing identifiers in the
report. If probabilities will be consumed as risk estimates, consider
a calibration curve and Brier score on held-out data; calibration is
optional for a pure ranking/alert task and must be fitted without
using final holdout or external labels. Feature importance or SHAP
may explain predictions but does not establish causality.

### 10.9 Required Figures and Reporting Rules

Generate and label, using **internal evaluation data**:

1. Confusion matrix with raw counts and threshold.
2. Precision-recall curve with AP and the evaluation prevalence line.
3. ROC curve with ROC-AUC where defined.
4. Threshold-versus-precision/recall/F1 and alert-count plot, or an
   alert-budget curve if top-K is the chosen operating policy.
5. Model/feature-group comparison and feature-importance or explanation
   figure for the selected model, with leakage risks reviewed.

Each figure and PDF table must identify the dataset partition,
positive class, score/threshold or `K`, and relevant denominators.
Keep training-only EDA, threshold-selection performance, untouched
holdout performance, and external predictions visibly distinct. Do
not claim external generalization performance without external
ground-truth labels. Save machine-readable metrics and the exact
configuration used to generate the report. Never fabricate results.

---

## 11. Model Interpretability

Include model interpretation where feasible.

Possible approaches:

- Feature importance
- Permutation importance
- SHAP analysis
- Logistic regression coefficients

Interpretations should answer:

- Which variables contribute most to predictions?
- Are these relationships plausible?
- Do the results align with EDA observations?
- Are there potential sources of leakage?
- Are the predictions explainable to stakeholders?

Do not interpret feature importance as causal evidence.

---

## 12. External Test Prediction

After final model selection:

1. Finalize preprocessing.
2. Finalize feature engineering.
3. Finalize model hyperparameters.
4. Finalize decision threshold.
5. Refit the selected pipeline on all eligible training data,
   when appropriate.
6. Load the untouched external test dataset.
7. Generate predictions.
8. Append `is_fraud_predicted`.
9. Save the output in Parquet format.

The output path must be:

`submission/test_fraud_external.parquet`

### Validation Checks

Before submission, verify:

```python
assert len(output_df) == len(original_test_df)

assert "is_fraud_predicted" in output_df.columns

assert output_df["is_fraud_predicted"].notna().all()

assert output_df["is_fraud_predicted"].isin([0, 1]).all()

assert output_df.columns.tolist() == (
    original_test_df.columns.tolist() + ["is_fraud_predicted"]
)
```

Also verify that all original values and row ordering
remain unchanged.

If the prediction column already exists in the source,
handle that situation explicitly rather than silently duplicating it.

Do not include prediction probabilities in the submitted Parquet
unless explicitly requested.

Probabilities may be saved separately for internal analysis.

---

## 13. PDF Report Requirements

The PDF must be professional, concise, and evidence-based.

Suggested filename:

`fraud_detection_report.pdf`

Suggested sections:

### 13.1 Executive Summary

Provide a brief, non-technical explanation of:

- The business problem
- The modeling approach
- The main findings
- The selected model
- Key validation results
- Practical implications

### 13.2 Dataset and EDA

Describe:

- Dataset characteristics
- Fraud prevalence
- Important distributions
- Missing values
- Notable patterns

Include informative visualizations.

### 13.3 Feature Engineering

Explain:

- Features created
- Motivation for each major feature category
- Data leakage prevention
- Expected usefulness

### 13.4 Modeling Methodology

Describe:

- Validation strategy
- Baseline models
- Candidate models
- Class imbalance handling
- Hyperparameter optimization
- Final model selection

### 13.5 Results

Include:

- Model comparison table
- Precision, Recall, F1
- PR-AUC
- Confusion matrix
- Precision-recall curve
- Feature importance
- Threshold analysis

Highlight the selected operating point.

### 13.6 Business Interpretation

Explain:

- How many fraudulent transactions are detected
- What false positives mean operationally
- What false negatives mean operationally
- Practical implications for a fraud investigation team

Avoid unsupported business impact claims.

### 13.7 Challenges and Limitations

Discuss actual challenges encountered.

Possible examples:

- Class imbalance
- Missing values
- High-cardinality categorical features
- Distribution shifts
- Limited historical information
- Potential data leakage
- Validation limitations
- Computational constraints
- False-positive costs
- Limited model interpretability

Do not claim a challenge occurred unless supported by analysis.

### 13.8 Future Improvements

Discuss realistic improvements such as:

- Additional transaction history
- Better behavioral features
- Temporal validation
- Entity-level modeling
- Graph/network analytics
- Improved probability calibration
- Cost-sensitive optimization
- More robust drift evaluation
- Production monitoring
- Human-in-the-loop review

Clearly separate completed work from future proposals.

---

## 14. Coding Standards

### 14.1 General Principles

Write clean, maintainable Python code.

Follow:

- PEP 8
- Meaningful variable names
- Functions with clearly defined responsibilities
- Type hints where useful
- Docstrings for important functions
- Relevant comments

Avoid unnecessary classes and complex abstractions.

### 14.2 Reproducibility

- Set random seeds.
- Record model hyperparameters.
- Save experiment configurations.
- Save evaluation metrics.
- Save trained model artifacts when useful.
- Keep data paths configurable.
- Ensure the pipeline can be executed again.

### 14.3 Logging

Use informative logging for major steps:

- Data loading
- Data validation
- Feature generation
- Model training
- Evaluation
- Prediction generation
- Output saving

Avoid excessive terminal output.

### 14.4 Error Handling

Handle:

- Missing input files
- Missing expected columns
- Unsupported data types
- Invalid target values
- Empty datasets
- Incompatible feature schemas
- Prediction failures
- Output writing errors

### 14.5 Data Integrity

Never silently modify the original input datasets.

Keep intermediate artifacts separate from raw data.

---

## 15. Execution Workflow

The coding agent should proceed in the following order.

### Phase 1 — Dataset Inspection

1. Inspect repository structure.
2. Verify Python environment.
3. Load column descriptions.
4. Inspect training data schema.
5. Identify target column.
6. Assess data quality.
7. Identify potential leakage risks.

Do not choose a complex model before understanding the data.

### Phase 2 — Exploratory Analysis

1. Perform initial EDA.
2. Identify class imbalance.
3. Analyze important variables.
4. Generate informative plots.
5. Document findings.

### Phase 3 — Baseline Modeling

1. Establish a suitable validation strategy.
2. Implement preprocessing.
3. Train baseline models.
4. Evaluate baseline performance.
5. Save baseline metrics.

### Phase 4 — Feature Engineering

1. Develop features supported by available data.
2. Prevent temporal and target leakage.
3. Evaluate feature usefulness.
4. Compare results with baseline.

### Phase 5 — Model Improvement

1. Train suitable candidate models.
2. Address class imbalance.
3. Tune hyperparameters when appropriate.
4. Compare models fairly.
5. Analyze errors.
6. Select the best validated solution.

### Phase 6 — Final Evaluation

1. Produce evaluation metrics.
2. Generate required plots.
3. Select the decision threshold.
4. Interpret model behavior.
5. Document limitations.

### Phase 7 — External Prediction

1. Finalize the selected pipeline.
2. Refit using eligible training data.
3. Generate external predictions.
4. Append `is_fraud_predicted`.
5. Validate output integrity.
6. Save the submission Parquet file.

### Phase 8 — Reporting

1. Summarize EDA discoveries.
2. Explain feature engineering.
3. Document modeling decisions.
4. Present actual validation results.
5. Include clear visualizations.
6. Discuss practical challenges.
7. Propose realistic improvements.
8. Generate the final PDF.

### Phase 9 — Submission Verification

Verify that all required deliverables exist.

Confirm that:

- Python code executes successfully.
- Results are reproducible.
- Figures match reported results.
- Metrics are correctly calculated.
- PDF contains all requested sections.
- Prediction Parquet has the correct schema.
- External data was not used for training.
- No additional prediction columns were introduced.
- Original external test data remains unchanged.

---

## 16. Critical Restrictions

The following rules are mandatory.

1. NEVER train on `test_fraud_external.parquet`.

2. NEVER use external test data for model selection,
   feature selection, preprocessing fitting, or threshold tuning.

3. NEVER use information unavailable at prediction time.

4. NEVER perform feature engineering that introduces
   target leakage or future information leakage.

5. NEVER evaluate model generalization using training
   performance alone.

6. NEVER optimize accuracy at the expense of meaningful
   fraud detection metrics without justification.

7. NEVER fabricate metrics, experimental results,
   visualizations, or analytical findings.

8. NEVER modify or overwrite the original datasets.

9. NEVER submit predictions with missing values.

10. NEVER change the row order or original column values
    of the external test dataset.

11. NEVER introduce unnecessary modeling complexity
    without demonstrating its usefulness.

12. ALWAYS ensure that the final project can be reproduced.

---

## 17. Priorities

When making technical decisions, use the following priorities:

1. Correctness and absence of data leakage
2. Valid and reliable model evaluation
3. Fraud detection performance
4. Reproducibility
5. Clear business interpretation
6. Readable and maintainable code
7. Computational efficiency
8. Additional model sophistication

A well-validated gradient boosting model is preferable to
an unnecessarily complex deep learning model without
demonstrated improvements.

The objective is not to use the most sophisticated algorithm.

The objective is to produce a reliable fraud detection solution
supported by rigorous experimental evidence.

---

## 18. Definition of Done

The assignment is complete only when:

- EDA has been performed.
- Informative plots have been generated.
- Feature engineering has been documented.
- A baseline model has been evaluated.
- A final model has been selected using valid evaluation.
- Precision, Recall, F1, and PR-AUC have been reported.
- Relevant evaluation plots have been generated.
- A decision threshold has been justified.
- Challenges and potential improvements have been discussed.
- A professional PDF report has been generated.
- External predictions have been generated.
- The required prediction column is present.
- All original external test rows and values are preserved.
- All deliverables have been verified.

The final submission must contain:

1. Python source code
2. Fraud detection PDF report
3. External test Parquet file with `is_fraud_predicted`

All results must be based on actual executions and observations.
