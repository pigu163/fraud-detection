# Fraud Detection Submission Code

This directory contains the Python implementation used for the exploratory
analysis, feature engineering, model comparison, validation, final model fit,
and external test prediction.

## Environment

The reported results were produced with Python 3.9.25. Install all required
Python packages from the project root:

```bash
python -m pip install -r submission/code/requirements.txt
```

The requirements file includes the notebook interface and kernel packages.
To open the exploratory notebook after installation, run:

```bash
python -m jupyter lab submission/code/EDA.ipynb
```

## Expected input files

Place the following unmodified files in the project root:

- `train_fraud.parquet`: labeled training transactions.
- `test_fraud_external.parquet`: unlabeled external transactions used only for
  final inference.
- `columns.txt`: field definitions supplied with the assignment.

The scripts accept command-line paths, so the data can be stored elsewhere if
the corresponding arguments are changed.

## Reproduce the analysis

Run the commands below from the project root in the listed order:

```bash
python submission/code/experiments/eda_summary.py --train train_fraud.parquet --output output
python submission/code/experiments/train_ablation.py --train train_fraud.parquet --test test_fraud_external.parquet --output output --submission-path submission/test_fraud_external.parquet
python submission/code/experiments/cv_3fold.py --train train_fraud.parquet --output output
python submission/code/experiments/temporal_validation.py --train train_fraud.parquet --output output/metrics
python submission/code/experiments/reference_rules.py --train train_fraud.parquet --holdout output/metrics/holdout_scores.parquet --output output/metrics/reference_rules.csv
python submission/code/experiments/report_figures.py --output output
```

`eda_summary.py` creates compact data-quality summaries and the principal EDA
figures. `EDA.ipynb` provides the fuller training-data analysis;
its reusable helpers are in `plot/` and `GFP/`.

`train_ablation.py` compares three classifiers across four feature groups on a
stratified fit/calibration/holdout split. It selects the model and threshold
using calibration rows, evaluates the frozen choice on the holdout, refits the
selected pipeline on all labeled training rows, and writes the binary external
predictions. The script preserves the external file's original index, row
order, columns, and values, and refuses to overwrite the source test file.

`cv_3fold.py` performs the three-fold stratified stability check.
`temporal_validation.py` runs the explicitly post-hoc chronological sensitivity
analysis. `reference_rules.py` evaluates the all-negative, supplied-flag, and
full-balance reference rules on the saved holdout positions. The figure script
uses saved internal metrics and the fitted model to generate report graphics.

## Output directories

- `output/metrics/`: model comparisons, holdout scores, cross-validation
  summaries, and sensitivity results.
- `output/figures/`: EDA and model-evaluation figures.
- `output/models/`: the selected fitted pipeline and its metadata.
- `output/predictions/`: external predictions and model scores.
- `submission/test_fraud_external.parquet`: final deliverable with the appended
  binary `is_fraud_predicted` column.

## Data isolation

All model selection, feature comparison, preprocessing fit, and threshold
selection use `train_fraud.parquet` only. The external test file is read only
after the candidate model and threshold have been finalized. Its distribution
and predictions are not used to revise the model.
