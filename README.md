# Fraud Detection

A Python machine learning project for detecting fraudulent financial transactions.
It covers exploratory data analysis, feature engineering, model comparison,
internal validation, and predictions for a separate external test dataset.

## Approach

- Explore class imbalance, transaction amounts, balances, and account activity.
- Compare Logistic Regression, XGBoost, and LightGBM across feature groups.
- Test amount-to-balance features and account/graph statistics constructed from
  strictly earlier hours.
- Select models using Average Precision (AP), choose a decision threshold on
  calibration data, and report fraud precision, recall, F1, and alert-budget metrics.
- Use stratified validation and cross-validation, with a separate post-hoc
  temporal sensitivity analysis.

The external test dataset is reserved for final inference. Pre-approval models
exclude post-transaction balances. Full-training-set EDA exposure and temporal
coverage differences limit how strongly internal validation can establish future
performance; see the report for details.

## Repository

| Path | Contents |
| --- | --- |
| `code/plot/` | EDA and visualization scripts |
| `code/GFP/` | Historical account and graph feature helpers |
| `code/experiments/` | Training, evaluation, and experiment scripts |
| `read_parquet_files.ipynb` | Exploratory notebook |
| `columns.txt` | Dataset field descriptions |
| `submission/code/` | Submission source, dependencies, and reproduction guide |
| `submission/fraud_detection_report.pdf` | Project report |

## Run

Use Python 3.9 and the dependencies in `submission/code/requirements.txt`.
Place the original `train_fraud.parquet` and `test_fraud_external.parquet` files
in the repository root, and read `columns.txt` before working with the data.

In the project's Windows environment, run from the repository root:

```powershell
& "D:\miniconda3\envs\nasdaq\python.exe" -m pip install -r submission/code/requirements.txt
& "D:\miniconda3\envs\nasdaq\python.exe" submission/code/experiments/eda_summary.py --train train_fraud.parquet --output output
& "D:\miniconda3\envs\nasdaq\python.exe" submission/code/experiments/train_ablation.py --train train_fraud.parquet --test test_fraud_external.parquet --output output --submission-path submission/test_fraud_external.parquet
```

See [the reproduction guide](submission/code/README.md) for the remaining
validation and reporting commands.

The prediction deliverable preserves the original external test rows and columns
and appends the binary `is_fraud_predicted` column. Datasets, model binaries,
generated output directories, and submission archives are excluded from Git.
