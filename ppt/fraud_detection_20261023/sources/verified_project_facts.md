# Fraud detection presentation source facts

Source boundary: the existing challenge repository. No new training or external-data model selection. Rechecked 2026-10-23. Slides must use English. User requests a concise black-background, white-text presentation saved under ppt/.

## Data and exploratory findings

- Unit: transaction. Positive label: is_fraud=1. Prediction time: before approval.
- output/metrics/eda_quality.json: 3,144,097 training rows, 11 columns including the target, 357 fraud, prevalence 0.0113546115%, no missing cells, duplicate rows, or constant raw fields.
- Five transaction types. Full-training EDA finds 250 fraud among 1,320,196 CASH_OUT and 107 among 337,569 TRANSFER. No observed fraud in other types is not proof of future safety.
- code/plot/full_balance_eda.py: A=transaction_amount, B=initiater_balance_before. Full-balance indicator is B>0 and abs(A-B)<=0.01. Full-training matches: 346/346 fraud, comprising 247 CASH_OUT and 99 TRANSFER. This covers 346/357 fraud. It is descriptive EDA with full-file discovery exposure, not independent validation precision.
- The 11 remaining fraud comprise 3 zero-amount, zero-starting-balance CASH_OUT and 8 TRANSFER. Avoid row-specific rules.
- Existing system flag: TP=2, FP=0, FN=355, TN=3,143,740. Precision=2/2, recall=2/357=0.5602%. A highly precise flag can still miss almost all fraud.
- Fraud hours: 635–743. Legitimate transactions end at 718. The 719–743 tail contains only fraud. Elapsed hours have no known calendar origin.
- High-cardinality account IDs are omitted as raw model inputs. Documented C/M prefixes provide role indicators.
- Same-pair prior count has no nonzero fraud examples. This alone does not establish irrelevance; the model ablation measures incremental value.
- Use 'strong observed association' and 'limited incremental evidence', not a formal ranking of feature sensitivity.

## Feature ablation and estimators

submission/code/experiments/train_ablation.py defines four groups, holding each estimator configuration fixed across groups:

1. raw: 10 columns. Hour, type, amount, two pre-transaction balances, supplied flag, four C/M role indicators.
2. raw_plus_extra: 23 columns. Add 13 amount/balance features: log amount, zero indicators, ratios, differences, overspend indicator, full-balance indicator, and type interactions.
3. raw_plus_graph: 22 columns. Add 12 earlier-hour graph/account statistics.
4. raw_plus_extra_plus_graph: 35 columns. Add both groups.

These counts precede one-hot encoding. Undefined ratios stay NaN, with training-partition median imputation and separate zero-balance indicators. Post-transaction balances and the target are excluded from predictors.

GFP/graph_features.py provides five strictly prior-hour features: outgoing/incoming counts, distinct counterparties, and directed pair count. gfp_additional_features.py adds six historical amount aggregates and reverse-edge count. This is a local GFP-inspired subset, not full GFP or a GNN. Same-hour rows never provide history to each other. The whole training transaction table supplies label-free history before random splitting, requiring an explicit online-history availability assumption.

Three baseline estimators: Logistic Regression (balanced, C=1, L-BFGS, max_iter=100), XGBoost (100 trees, depth=4, learning_rate=.08, subsampling=.8), LightGBM (100 trees, 31 leaves, learning_rate=.08, min_child_samples=20). Boosting uses the fit-partition negative/positive ratio as scale_pos_weight. No hyperparameter search or SMOTE.

Numeric preprocessing: median imputation and StandardScaler. Transaction type: most-frequent imputation and OneHotEncoder(handle_unknown='ignore'). Pipeline fitting stays within model training rows.

## Main partition and selection

Two nested StratifiedShuffleSplit calls, seed 42: outer holdout 20%, then calibration 25% of the remaining 80%. Overall fit/calibration/holdout proportions are 60/20/20.

| Partition | Rows | Fraud | Role |
|---|---:|---:|---|
| Fit | 1,886,457 | 214 | Fit preprocessing and estimator |
| Calibration | 628,820 | 72 | Compare AP and select F1 threshold |
| Holdout | 628,820 | 71 | Internal evaluation |

Candidates rank by calibration AP, calibration F1, and fewer input columns. Fully tied candidates retain iteration order. Threshold maximizes fraud-class calibration F1. Calibration is a data partition, not probability calibration. Code saves holdout outcomes for all candidates; selection logic uses calibration only. Full-file EDA preceded splitting.

## Saved experimental results

Authoritative source: output/metrics/model_metrics.csv, not discrepant PDF numbers.

| Feature group | Estimator | Holdout AP | TP | FP | FN | F1 |
|---|---|---:|---:|---:|---:|---:|
| raw | Logistic Regression | .190728702 | 10 | 5 | 61 | .232558140 |
| raw | XGBoost | .776467600 | 46 | 17 | 25 | .686567164 |
| raw | LightGBM | .008153422 | 71 | 8,637 | 0 | .016174963 |
| raw_plus_extra | Logistic Regression | 1 | 71 | 0 | 0 | 1 |
| raw_plus_extra | XGBoost | 1 | 70 | 0 | 1 | .992907801 |
| raw_plus_extra | LightGBM | .000319326 | 68 | 206,976 | 3 | .000656640 |
| raw_plus_graph | Logistic Regression | .167248302 | 7 | 7 | 64 | .164705882 |
| raw_plus_graph | XGBoost | .760550988 | 39 | 7 | 32 | .666666667 |
| raw_plus_graph | LightGBM | .021690224 | 69 | 3,023 | 2 | .043629466 |
| raw_plus_extra_plus_graph | Logistic Regression | 1 | 71 | 0 | 0 | 1 |
| raw_plus_extra_plus_graph | XGBoost | 1 | 70 | 0 | 1 | .992907801 |
| raw_plus_extra_plus_graph | LightGBM | .001494517 | 71 | 47,436 | 0 | .002984573 |

Selected: raw_plus_extra Logistic Regression. 23 input columns, threshold .9992964880111456. Holdout TP=71, FP=0, FN=0, TN=628,749. AP=1 and ROC-AUC=1 in saved main metrics. Graph did not improve this selected combination. LightGBM failure concerns the tested fixed configuration, not the entire algorithm family.

Reference full-balance rule on the same holdout: TP=70, FP=0, FN=1. AP=.985917083. One additional model detection is not independent proof because the rule discovery used full-file labels.

## Three-fold stability check

submission/code/experiments/cv_3fold.py uses StratifiedKFold(n_splits=3, shuffle=True, random_state=42). Only raw and raw_plus_extra are checked, for all three estimators. Every outer training fold gets an inner stratified 80/20 fit/calibration split to choose the threshold. Then refit on the complete outer-training fold and evaluate the held-out outer fold. Threshold transfer after refitting can change the score scale.

output/metrics/cv3fold_summary.csv:

| Group | Estimator | Mean AP | AP SD | Mean recall | Mean F1 | F1 SD |
|---|---|---:|---:|---:|---:|---:|
| raw | Logistic Regression | .252038579 | .008985322 | .229691877 | .344586168 | .038539863 |
| raw_plus_extra | Logistic Regression | .999976657 | .000040431 | .983193277 | .991489362 | .007370429 |
| raw | XGBoost | .765391227 | .027646510 | .529411765 | .662970816 | .047617063 |
| raw_plus_extra | XGBoost | .999256493 | .001287792 | .980392157 | .989972841 | .013874766 |

Interpretation: engineered feature gains persist across random folds. This reduces dependence on a single random partition, but does not remove discovery exposure, shared-entity concerns, or future temporal drift.

## Metrics: purpose and calculation

- AP is primary for ranking under rare positives. AP=sum over thresholds of (recall increment * precision), as sklearn.metrics.average_precision_score. Evaluate continuous fraud scores. This is not trapezoidal PR area. No-skill precision is the same evaluation partition's prevalence.
- Precision=TP/(TP+FP): investigation quality.
- Recall=TP/(TP+FN): fraud coverage.
- F1=2TP/(2TP+FP+FN): balances precision and recall for operating-threshold selection when no costs are supplied.
- ROC-AUC=P(score of a random fraud exceeds score of a random legitimate)+0.5*P(tie). Secondary ranking context. High ROC-AUC can coexist with poor precision.
- FPR=FP/(FP+TN), with absolute FP count always visible.
- Alerts=TP+FP. Alert rate=(TP+FP)/N. False alerts per 100,000=100000*FP/N.
- Always-negative accuracy on full training is 99.9886%, despite zero recall.
- No Precision@K/Recall@K results were saved. Do not imply those experiments ran or invent an operational alert budget.
- Scores from class-weighted estimators are not calibrated fraud probabilities. The submitted operating policy is a fixed threshold, not top-K.

## Reliability and final submission

Post-hoc temporal sensitivity fits hours 0–659, selects threshold on 660–680, evaluates 681–718 (14,939 rows, 64 fraud). Engineered logistic regression detects 64/64 with 0 FP, including a variant removing hour and flag together. Boundaries followed observed label coverage, so this is not a blind future test.

Full-training refit precedes external inference in submission code. Metadata records 2,247,484 external rows with 829 positive predictions. Prediction prevalence is not measured external fraud prevalence or accuracy. The submission consists of code, PDF report, and external Parquet with binary is_fraud_predicted. Historical results used Python 3.9.25; current Python 3.12.15 environment has not rerun the complete experiments.

## Available figure sources

image/ contains 25 PDF figures. Particularly useful: label_distribution, full_balance_rule_by_type, amount_to_starting_balance_bins, system_flag_performance, time_distribution_training, gfp_target_prior_in_fan_hours_635_718. Model figure PDFs and saved CSV results live under output/. Reuse original evidence or rebuild editable data charts from the saved numerical sources. Preserve scientific meaning and sample denominators when adapting to dark slides.
