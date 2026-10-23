# Fraud Detection Challenge

Transaction-level fraud detection take-home project. The repository contains the
Python pipeline, exploratory analysis, an internal validation study, and the
submitted PDF report. The external test set is used only to generate predictions.

## Results at a glance

The labeled training file contains 3,144,097 transactions, including 357 fraud
cases (0.0114%). We compared Logistic Regression, XGBoost, and LightGBM across
four feature groups: raw transaction fields; raw plus amount/balance features;
raw plus earlier-hour account/graph history; and all three groups combined.

The selected class-weighted Logistic Regression uses raw plus amount/balance
features. On the internal stratified holdout (628,820 rows, 71 fraud), its
Average Precision (AP) was 1.000. At a threshold selected on a separate
calibration partition, it found 71/71 fraud cases with zero false alerts. A
three-fold stratified check gave mean AP 0.99998 and mean fraud-class F1 0.9915.
A separate *post-hoc* chronological sensitivity check is also included.

These are within-file results. The strongest balance pattern was discovered
using labels from the full training file, fraud occurs in a restricted late
time range, and the chronological boundaries were chosen after inspecting
label coverage. The results are **not a prospective performance estimate**.
See the [report](submission/fraud_detection_report.pdf) for the full evidence
and limitations.

## Repository map

| Path | Contents |
| --- | --- |
| [`submission/code/`](submission/code/) | Submitted implementation, notebook, requirements, and [reproduction guide](submission/code/README.md) |
| [`submission/fraud_detection_report.pdf`](submission/fraud_detection_report.pdf) | Final written report |
| [`ppt/Fraud_Detection_Case_Study_Latest.pptx`](ppt/Fraud_Detection_Case_Study_Latest.pptx) | Editable presentation; its SVG source and notes are in [`ppt/fraud_detection_20261023/`](ppt/fraud_detection_20261023/) |
| [`image/`](image/) | 25 EDA and graph-analysis PDF figures |
| [`coaching_state.md`](coaching_state.md) | 30 project-specific interview questions with evidence-based short answers |
| [`code/`](code/) | Development EDA and graph-feature scripts |
| [`columns.txt`](columns.txt) | Supplied field descriptions |

## Reproduce

Use the existing `nasdaq` Conda environment (Python 3.12.15). On this Windows
machine its interpreter is `D:\miniconda3\envs\nasdaq\python.exe`; install the
pinned packages in [`submission/code/requirements.txt`](submission/code/requirements.txt).
The reported experiments were originally run with Python 3.9.25 and have **not**
been rerun after the environment migration.

Place the unmodified `train_fraud.parquet` and `test_fraud_external.parquet` in
the repository root, and consult `columns.txt` before interpreting fields. From
the repository root, run:

```powershell
& "D:\miniconda3\envs\nasdaq\python.exe" -m pip install -r submission/code/requirements.txt
& "D:\miniconda3\envs\nasdaq\python.exe" submission/code/experiments/eda_summary.py --train train_fraud.parquet --output output
& "D:\miniconda3\envs\nasdaq\python.exe" submission/code/experiments/train_ablation.py --train train_fraud.parquet --test test_fraud_external.parquet --output output --submission-path submission/test_fraud_external.parquet
```

The [reproduction guide](submission/code/README.md) lists the cross-validation,
chronological check, reference-rule, and report-figure commands. Preprocessing
is fit only on training partitions. The model excludes post-transaction balances
for its pre-approval prediction setting. Graph features use strictly earlier
elapsed hours; same-hour transaction order is unknown.

The prediction deliverable appends binary `is_fraud_predicted` to every original
external row while preserving its order and columns. The source external file
is not overwritten. Local Parquet inputs and outputs, model binaries, generated
metrics, and submission archives are excluded from Git.
