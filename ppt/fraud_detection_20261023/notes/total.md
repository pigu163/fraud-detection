## 01_transaction_fraud_detection

This project detects fraud at the individual transaction level before approval. I will show how exploratory analysis led to four feature configurations, how three baseline classifiers compared, and how I assessed the unusually high internal scores. The main measured improvement comes from expressing the relationship between transaction amount and starting balance. The final submission uses a simple weighted logistic regression, with clear limits on what the current validation can establish. The evidence comes from the submitted Python code, saved experiment CSVs and training-only EDA figures in this challenge repository.

## 02_rare_fraud_usable_records

The training file contains 3,144,097 transactions and only 357 fraud labels, about 0.0114 percent. Basic record-level checks found no missing cells, duplicate records or constant raw fields. There are ten original input fields, the target, and five transaction types. These checks make the records usable, but they do not establish that the labels are representative or that the data are suitable for a prospective test. Our prediction unit is a transaction and the positive class is is_fraud equal to one. I exclude post-transaction balances because a pre-approval decision cannot observe them. The source for these counts is output/metrics/eda_quality.json, with field definitions in columns.txt.

## 03_strongest_observed_relationship

The clearest observed relationship is between amount and the initiator's balance before the transaction. I define a full-balance match as a strictly positive starting balance and an amount within one cent of that balance. All 346 matching transactions in the full training file carry fraud labels. They account for 346 of the 357 fraud cases: 247 cash-outs and 99 transfers. The original EDA figure compares those rule matches with the many other transactions in each type, with a logarithmic transaction-count axis. The remaining eleven comprise three zero-amount, zero-starting-balance cash-outs and eight transfers. This motivates ratios, balance differences, zero indicators and interactions with transaction type rather than special rules for individual rows. This is a descriptive finding discovered with all training labels visible, so the later holdout is exposed to the feature hypothesis. The relevant implementation is code/plot/full_balance_eda.py, and the original figure is image/交易金额与余额行为/full_balance_rule_by_type.pdf.

## 04_useful_context_limited_signals

Transaction type provides useful context. The observed fraud consists of 250 cash-outs among 1,320,196 cash-outs and 107 transfers among 337,569 transfers. Their fraud rates are 0.0189 and 0.0317 percent respectively. The absence of fraud in the other types is a sample finding, not a rule about future safety. I also inspected signals with limited standalone coverage. The supplied system flag identifies only two of the 357 fraud cases with no observed false positives, giving two-over-two precision but only 0.56 percent recall. Prior same-pair count is nonzero in none of the fraud cases, so there is little positive variation for this feature. I would not claim a universal sensitivity ranking from these observations. Incremental feature value needs the model ablation. Sources are transaction_type_eda.py, system_flag_eda.py and the GFP feature summaries under outputs/metrics.

## 05_temporal_coverage

EDA: fraud coverage starts at hour 635; legitimate records end at 718. This motivates the later supplementary chronological check.

## 06_four_feature_groups

I compare four feature groups while keeping each estimator's configuration fixed. Raw contains ten columns: elapsed hour, type, amount, the two starting balances, the supplied flag and four customer-role indicators from documented C and M prefixes. The extra group adds thirteen features derived from amount and starting balances, including ratios, differences, zero indicators and full-balance interactions. The graph group adds twelve historical account and pair statistics. That gives ten, twenty-three, twenty-two and thirty-five inputs before one-hot encoding. Ratios are undefined when the denominator is not positive and use an imputer learned only on fitting rows, with zero balance separately indicated. Graph histories use strictly earlier hours and never same-hour transactions or labels. The history table is built before the random split, so its validity depends on access to earlier unlabeled transactions at scoring time. This is a local GFP-inspired subset, not full GFP or a graph neural network. Sources are CONFIGS in the submitted train_ablation.py and the two GFP feature builders.

## 07_tabular_feature_inventory

Complete semantic inventory: Raw 10 plus Extra 13. Both zero-balance flags refer to starting balances. Full-balance type interactions multiply the matching rule by TRANSFER and CASH_OUT indicators. All 13 additions are defined without target labels; hypothesis discovery nevertheless used full-training EDA.

## 08_graph_feature_inventory

Graph 12 is five count features, six amount statistics and one reverse-edge count. In this implementation degree means transaction count, while fan means number of distinct counterparties. Reciprocal cycle count is earlier reverse-edge frequency, not an arbitrary cycle search. Features use hours strictly less than the current hour. Graph features are label-free but the random protocol assumes available earlier unlabeled history across partitions. Their value is assessed by ablation; they did not improve the selected Extra model.

## 09_main_stratified_split

The main study uses two nested StratifiedShuffleSplit calls with random seed forty-two. First, twenty percent becomes holdout. Then twenty-five percent of the remaining eighty percent becomes calibration, which is twenty percent of the original data. The final proportions are sixty percent fit, twenty percent calibration and twenty percent holdout. Fit has 1,886,457 rows and 214 fraud cases. Calibration has 628,820 rows and 72 fraud cases. Holdout has the same row count and 71 fraud cases. Stratification preserves the rare-positive proportions and prevents a purely random allocation from putting too few positives in one partition. It does not enforce chronology or separate shared account entities. Preprocessing and classifier fitting use fit rows; candidate and threshold selection use calibration rows. Full-file EDA still limits the independence of the eventual holdout. The source is the run function in submitted train_ablation.py.

## 10_model_and_threshold_selection

For each candidate, I fit its pipeline on fit rows and obtain calibration scores. I choose its threshold by maximizing calibration fraud-class F1. Candidates then rank by calibration AP, followed by calibration F1 and fewer input columns. When these values fully tie, the fixed iteration order retains the earlier candidate, which is why choosing logistic regression over an equally scoring XGBoost is not proof that logistic regression is universally better. The script computes and saves holdout results for all candidates, but its selection branch uses calibration only. After the chosen operating policy is frozen, the final selected pipeline refits on all training rows and performs external inference. Calibration here names a data partition; no probability calibration was performed. A score near one from a class-weighted model cannot be read as a matching real-world fraud probability. Reusing calibration for model comparison and threshold choice can introduce optimism. Refitting can also shift the score scale while the frozen threshold stays fixed. The source is best_f1_threshold and ranking_key in the submitted training script.

## 11_three_baselines

I use logistic regression as an interpretable linear baseline and two tree-based baselines to capture nonlinear interactions. Logistic regression has balanced class weights, C equal to one and the L-BFGS solver. XGBoost uses one hundred trees, depth four and learning rate 0.08. LightGBM uses one hundred trees, thirty-one leaves and learning rate 0.08. The boosting models use the negative-to-positive ratio from the fitting partition as their positive-class weight. All candidates use numeric median imputation and standardization, plus most-frequent imputation and one-hot encoding for transaction type. The pipeline fits these transformations only on its model-fitting rows. I use no resampling and no hyperparameter search. The purpose is a controlled feature comparison rather than a claim that each algorithm has been fully optimized. The exact settings are in make_model and make_preprocessor in submission/code/experiments/train_ablation.py.

## 12_ablation_results

This table reports Average Precision on the same 628,820-row holdout, containing 71 fraud cases. Raw logistic regression reaches AP 0.1907 and raw XGBoost reaches 0.7765. Adding amount and balance relationships raises both to AP one. Adding graph history to the selected engineered logistic model produces no measurable gain here. Graph history alone also does not improve raw XGBoost AP. LightGBM performs poorly under this fixed configuration, including many false alerts and saturated scores. That is a result about the tested settings, not proof that the algorithm family is unsuitable for fraud. Candidate selection itself uses calibration AP, with calibration F1 and smaller feature count as tie-breakers. The numerical authority is output/metrics/model_metrics.csv. In particular, the raw-plus-extra XGBoost holdout AP is one in that CSV, whereas its three-fold mean AP is about 0.99926.

## 13_selected_operating_point

The selected candidate is logistic regression with twenty-three raw and engineered inputs. Its calibration-selected threshold is approximately 0.999296488. On the internal holdout it produces 71 true positives, no false positives or false negatives, and 628,749 true negatives. This means precision, recall and fraud-class F1 are all one, with 71 alerts among 628,820 transactions. The full-balance rule alone detects seventy of the seventy-one fraud cases without false alerts. The model recovers one additional case in this partition, but this is not an independent confirmation of the discovery because the rule itself came from full-file labels. Only seventy-one positive observations support the displayed perfect rates, so I would not promise future zero error. Sources are output/metrics/model_metrics.csv, reference_rules.csv and output/models/selected_model_metadata.json.

## 14_ranking_metrics

AP summarizes precision as recall grows. The project uses sklearn average_precision_score on continuous scores, with fraud positive. Formula: AP = sum((R_n - R_previous) * P_n). This is not trapezoidal PR area. ROC-AUC measures pairwise ranking, counting tied scores as half. AP is primary under extreme imbalance; all-negative accuracy is unhelpful.

## 15_threshold_metrics

Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = 2PR/(P+R). F1 is the harmonic mean, not an ordinary arithmetic average, and stays low when either component is low. Example is illustrative: 8 TP, 2 FP, 2 FN; P=R=F1=0.8. Threshold selection used calibration labels only.

## 16_three_fold_validation

The stability check is separate from the main shuffle split. It uses StratifiedKFold with three folds, shuffling and random seed forty-two. In each outer round, one fold is validation and the other two form the outer training data. Inside that outer training set, a stratified eighty-twenty split supplies model-fitting and calibration rows to choose the threshold. The classifier then refits on the complete outer training set and evaluates the unseen outer fold. Thus the outer validation labels do not select their threshold. Refitting can change score scale, so the threshold transfer is itself a practical limitation. The check compares only raw and raw-plus-extra features across the three estimators. It does not validate the graph configurations. A, B and C are explanatory labels for the fold roles. The authoritative implementation is submission/code/experiments/cv_3fold.py.

## 17_cross_validation_results

The engineered features retain their improvement across the three random folds. The bar chart uses a common zero-to-one Average Precision scale, with white bars for engineered inputs and gray bars for raw inputs. Logistic regression moves from mean AP about 0.25204 with raw inputs to 0.99998 with the engineered group. Its AP standard deviation is about 0.00004, and mean F1 is 0.99149 with standard deviation 0.00737. XGBoost also improves from mean AP 0.76539 to 0.99926, with mean F1 rising from 0.66297 to 0.98997. The selected logistic model has mean precision one and mean recall 0.98319, so its single-holdout perfect recall does not repeat in every fold. These findings reduce dependence on one fortunate random partition and support the feature group's value within this file. They do not eliminate the prior label exposure involved in discovery, nor test future temporal or new-entity generalization. Full saved results also contain LightGBM. All numbers here come from output/metrics/cv3fold_summary.csv, rounded for presentation.

## 18_temporal_coverage

Chronological sensitivity check using training data only. Fit on hours 0-659: 3,126,823 rows and 215 fraud. Choose the fraud-class F1 threshold on hours 660-680: 2,294 rows and 37 fraud. Keep the fitted model and threshold fixed, without refitting, and evaluate on hours 681-718: 14,939 rows and 64 fraud. Hours 719-743 contain 41 fraud and no legitimate transactions, so they only support an additional fraud-detection check. The boundaries were selected after observing labeled coverage; this is supplementary post-hoc evidence. Segment widths are schematic, not proportional to row counts or duration.

## 19_validation_limits

Compare the three Logistic Regression variants under the same chronological split. Raw finds all 64 fraud but issues 8,452 false alerts. Adding Extra finds all 64 with zero false alerts; removing elapsed hour and the system flag together preserves that result. AP and F1 are shown separately because AP measures ranking while F1 uses the frozen calibration threshold. Boundaries were chosen after observing labeled coverage and feature discovery used full-file labels. Only 64 positive cases are evaluated. Random folds can share entities; the system flag timing needs business confirmation. A future untouched mixed-class period remains the next test.


## 20_final_inference

For final delivery, the selected logistic pipeline refits on all 3,144,097 labeled training rows. The submitted code reads the external file only after model selection and full-training fitting, applies the same feature definitions and the frozen threshold, and appends the binary is_fraud_predicted column. It preserves the original rows, order, index and column values. The output contains 829 positive predictions among 2,247,484 external transactions. This is a prediction count, not a measured count of fraud or an external accuracy estimate, because external labels are unavailable. The final challenge deliverables are Python code, a PDF report and the prediction Parquet. Historical experiments used Python 3.9.25; the current Python 3.12.15 environment has not rerun the entire study. The main lesson is that useful feature relationships let a simple model perform strongly within this file, while prospective evidence is still needed before operational deployment. Sources are selected_model_metadata.json and the final inference branch of the submitted train_ablation.py.
