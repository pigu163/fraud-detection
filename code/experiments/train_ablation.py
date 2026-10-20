# Academic Figure Skill Asset Confirmation (verified against existing project figures)
# (a) model AP comparison → quantitative_grid → param inherit
# (b) precision–recall curves → quantitative_grid → param inherit
# (c) threshold selection → quantitative_grid → param inherit
# RULE: all panels are drawn directly below with the copied journal-style baseline.

"""Small, leakage-aware ablation study for the fraud challenge.

The script trains four feature configurations with three binary classifiers,
selects a model and threshold on an internal calibration split, evaluates on
an untouched stratified holdout, then writes external predictions.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib as mpl

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette — COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM, PDF-only per user request
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from matplotlib.text import Text
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
GFP_DIR = ROOT / "code" / "GFP"
sys.path.insert(0, str(GFP_DIR))
from gfp_additional_features import build_additional_gfp_features  # noqa: E402
from graph_features import build_gfp_features  # noqa: E402


TRAIN_DEFAULT = ROOT / "train_fraud.parquet"
TEST_DEFAULT = ROOT / "test_fraud_external.parquet"
OUTPUT_DEFAULT = ROOT / "output"
RANDOM_STATE = 42

# ponytail: omit high-cardinality account IDs; add them only if validation
# shows a reproducible benefit from a leakage-safe encoding.
# Raw inputs deliberately omit the high-cardinality account IDs. Their C/M
# role indicators are added below; one-hot encoding millions of IDs is not a
# useful baseline for this transaction-level task.
RAW_COLUMNS = [
    "hours_elapsed",
    "transaction_type",
    "transaction_amount",
    "initiater_balance_before",
    "recipient_balance_before",
    "is_flagged_fraud",
    "initiating_customer_is_C",
    "initiating_customer_is_M",
    "recipient_customer_is_C",
    "recipient_customer_is_M",
]
EXTRA_COLUMNS = [
    "transaction_amount_log1p",
    "initiater_balance_before_is_zero",
    "recipient_balance_before_is_zero",
    "amount_to_initiater_balance_ratio",
    "amount_to_recipient_balance_ratio",
    "amount_minus_initiater_balance",
    "abs_amount_minus_initiater_balance",
    "amount_exceeds_initiater_balance",
    "is_full_balance_amount_match",
    "transaction_amount_is_zero",
    "amount_and_initiater_balance_are_zero",
    "is_transfer_and_full_balance",
    "is_cash_out_and_full_balance",
]
GRAPH_COLUMNS = [
    "gfp_source_prior_out_degree",
    "gfp_source_prior_out_fan",
    "gfp_target_prior_in_degree",
    "gfp_target_prior_in_fan",
    "gfp_pair_prior_count",
    "gfp_source_prior_out_amount_sum",
    "gfp_source_prior_out_amount_mean",
    "gfp_source_prior_out_amount_max",
    "gfp_target_prior_in_amount_sum",
    "gfp_target_prior_in_amount_mean",
    "gfp_target_prior_in_amount_max",
    "gfp_prior_reciprocal_cycle_count",
]
CONFIGS = {
    "raw": RAW_COLUMNS,
    "raw_plus_extra": RAW_COLUMNS + EXTRA_COLUMNS,
    "raw_plus_graph": RAW_COLUMNS + GRAPH_COLUMNS,
    "raw_plus_extra_plus_graph": RAW_COLUMNS + EXTRA_COLUMNS + GRAPH_COLUMNS,
}
ALGORITHMS = ("logistic_regression", "xgboost", "lightgbm")
GRAPH_INPUT_COLUMNS = [
    "hours_elapsed",
    "initiating_customer",
    "recipient_customer",
    "transaction_amount",
]


def save_pdf(fig: plt.Figure, path: Path) -> Path:
    """Save a vector PDF only; the user requested no PNG previews."""
    for artist in fig.findobj(match=Text):
        if artist.get_text():
            artist.set_fontsize(max(7, artist.get_fontsize()))
    fig.tight_layout(pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def add_role_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add documented customer-role indicators without using the target."""
    result = df.copy()
    for role in ("initiating_customer", "recipient_customer"):
        values = result[role].astype("string")
        result[f"{role}_is_C"] = values.str.startswith("C").astype("int8")
        result[f"{role}_is_M"] = values.str.startswith("M").astype("int8")
    return result


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """Return NaN for undefined zero-denominator ratios."""
    return np.divide(
        numerator.to_numpy(dtype=float),
        denominator.to_numpy(dtype=float),
        out=np.full(len(numerator), np.nan, dtype=float),
        where=denominator.to_numpy(dtype=float) > 0,
    )


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create pre-transaction amount and balance features."""
    result = add_role_features(df)
    amount = result["transaction_amount"]
    initiator_before = result["initiater_balance_before"]
    recipient_before = result["recipient_balance_before"]
    full_balance = (initiator_before > 0) & ((amount - initiator_before).abs() <= 0.01)
    result["transaction_amount_log1p"] = np.log1p(amount.clip(lower=0))
    result["initiater_balance_before_is_zero"] = initiator_before.eq(0).astype("int8")
    result["recipient_balance_before_is_zero"] = recipient_before.eq(0).astype("int8")
    result["amount_to_initiater_balance_ratio"] = safe_ratio(amount, initiator_before)
    result["amount_to_recipient_balance_ratio"] = safe_ratio(amount, recipient_before)
    result["amount_minus_initiater_balance"] = amount - initiator_before
    result["abs_amount_minus_initiater_balance"] = (amount - initiator_before).abs()
    result["amount_exceeds_initiater_balance"] = (amount > initiator_before).astype("int8")
    result["is_full_balance_amount_match"] = full_balance.astype("int8")
    result["transaction_amount_is_zero"] = amount.eq(0).astype("int8")
    result["amount_and_initiater_balance_are_zero"] = (
        amount.eq(0) & initiator_before.eq(0)
    ).astype("int8")
    result["is_transfer_and_full_balance"] = (
        result["transaction_type"].eq("TRANSFER") & full_balance
    ).astype("int8")
    result["is_cash_out_and_full_balance"] = (
        result["transaction_type"].eq("CASH_OUT") & full_balance
    ).astype("int8")
    return result


def build_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the existing strict-prior-hour graph feature set."""
    graph_input = df.loc[:, GRAPH_INPUT_COLUMNS].reset_index(drop=True)
    basic = build_gfp_features(graph_input)
    amount = build_additional_gfp_features(graph_input)
    return pd.concat([basic, amount], axis=1).loc[:, GRAPH_COLUMNS]


def make_feature_frames(df: pd.DataFrame, graph: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create the four requested ablation matrices."""
    enriched = add_extra_features(df)
    raw = enriched.loc[:, RAW_COLUMNS].copy()
    extra = enriched.loc[:, EXTRA_COLUMNS].copy()
    graph = graph.reset_index(drop=True)
    return {
        "raw": raw,
        "raw_plus_extra": pd.concat([raw, extra], axis=1),
        "raw_plus_graph": pd.concat([raw, graph], axis=1),
        "raw_plus_extra_plus_graph": pd.concat([raw, extra, graph], axis=1),
    }


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in columns if column == "transaction_type"]
    numeric = [column for column in columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def make_model(name: str, positive_weight: float):
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=100, solver="lbfgs", n_jobs=8
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=positive_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=8,
            random_state=RANDOM_STATE,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=100,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=positive_weight,
            objective="binary",
            verbosity=-1,
            n_jobs=8,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown algorithm: {name}")


def make_pipeline(columns: list[str], algorithm: str, positive_weight: float) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(columns)),
            ("model", make_model(algorithm, positive_weight)),
        ]
    )


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Select one threshold using calibration labels only."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    index = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[index]) if len(thresholds) else 0.5


def evaluate_scores(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predicted = (scores >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan,
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "alert_count": int(predicted.sum()),
        "alert_rate": float(predicted.mean()),
        "false_alerts_per_100k": float(100_000 * fp / len(y_true)),
    }


def plot_model_comparison(metrics: pd.DataFrame, path: Path) -> None:
    plot_data = metrics.sort_values(["feature_config", "algorithm"])
    labels = [f"{row.feature_config}\n{row.algorithm}" for row in plot_data.itertuples()]
    colors = [CATEGORICAL[ALGORITHMS.index(name)] for name in plot_data.algorithm]
    fig, ax = plt.subplots(figsize=(183 / 25.4, 105 / 25.4))
    bars = ax.bar(np.arange(len(plot_data)), plot_data["holdout_ap"], color=colors, width=0.72)
    ax.set_xticks(np.arange(len(plot_data)), labels, rotation=55, ha="right")
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title("Ablation comparison on the untouched stratified holdout", fontweight="bold")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, plot_data["holdout_ap"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                ha="center", va="bottom", fontsize=6, rotation=90)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORICAL[i]) for i in range(3)]
    ax.legend(handles, [name.replace("_", " ").title() for name in ALGORITHMS], loc="upper left")
    fig.text(0.5, 0.005, "Bars show fraud-score AP; threshold-dependent metrics use a calibration-selected F1 threshold.",
             ha="center", fontsize=7.5)
    save_pdf(fig, path)


def plot_precision_recall_curves(
    y_true: np.ndarray, holdout_scores: dict[tuple[str, str], np.ndarray], metrics: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 130 / 25.4), sharex=True, sharey=True)
    prevalence = float(y_true.mean())
    for ax, config in zip(axes.flat, CONFIGS):
        for algorithm, color in zip(ALGORITHMS, CATEGORICAL[:3]):
            precision, recall, _ = precision_recall_curve(y_true, holdout_scores[(config, algorithm)])
            ap = metrics.loc[
                (metrics.feature_config == config) & (metrics.algorithm == algorithm), "holdout_ap"
            ].iloc[0]
            ax.plot(recall, precision, color=color, linewidth=1.1, label=f"{algorithm.replace('_', ' ').title()} (AP={ap:.3f})")
        ax.axhline(prevalence, color=GREY, linestyle="--", linewidth=0.6, label="Prevalence")
        ax.set_title(config.replace("_", " + "), fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(color="#E0E0E0", linewidth=0.3, alpha=0.6)
    axes[0, 0].legend(loc="upper right", fontsize=6)
    fig.suptitle("Precision–recall curves on the same holdout", fontweight="bold")
    fig.text(0.5, 0.005, f"Positive class: is_fraud=1; holdout prevalence={prevalence:.5f}; dashed line is no-skill precision.",
             ha="center", fontsize=7.5)
    save_pdf(fig, path)


def plot_threshold_selection(
    y_true: np.ndarray, scores: np.ndarray, selected_threshold: float, path: Path
) -> None:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.50, 1.0, 160)))
    rows = []
    for threshold in thresholds:
        values = evaluate_scores(y_true, scores, float(threshold))
        rows.append({"threshold": threshold, "precision": values["precision"],
                     "recall": values["recall"], "f1": values["f1"],
                     "alerts": values["alert_count"]})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 88 / 25.4))
    ax.plot(table.threshold, table.precision, color=CATEGORICAL[0], linewidth=1.1, label="Precision")
    ax.plot(table.threshold, table.recall, color=CATEGORICAL[1], linewidth=1.1, label="Recall")
    ax.plot(table.threshold, table.f1, color=CATEGORICAL[2], linewidth=1.1, label="F1")
    ax.axvline(selected_threshold, color=BLACK, linestyle="--", linewidth=0.8,
               label=f"Selected threshold={selected_threshold:.5g}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.05)
    ax.set_title("Threshold selection on the calibration split", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.3, alpha=0.6)
    fig.text(0.5, 0.005, "The threshold maximizes fraud-class F1 on calibration labels; the final holdout is not used here.",
             ha="center", fontsize=7.5)
    save_pdf(fig, path)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    submission_path = Path(args.submission_path)
    if submission_path.resolve() == Path(args.test).resolve():
        raise ValueError("Submission path must differ from the original external input")
    metrics_dir = output / "metrics"
    figures_dir = output / "figures"
    models_dir = output / "models"
    predictions_dir = output / "predictions"
    for directory in (metrics_dir, figures_dir, models_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Load both files up front. The external rows are kept isolated and are
    # not used for fitting, feature selection, threshold selection or scoring.
    train = pd.read_parquet(args.train).reset_index(drop=True)
    external_with_index = pd.read_parquet(args.test)
    # Keep the source Parquet's pandas row index as part of the delivered file.
    # Feature matrices use positional indexes so graph columns align correctly.
    external_index = external_with_index.index.copy()
    external = external_with_index.reset_index(drop=True)
    if "is_fraud" not in train.columns:
        raise ValueError("Training data must contain is_fraud")
    if not train["is_fraud"].isin([0, 1]).all():
        raise ValueError("is_fraud must be binary")
    y = train["is_fraud"].to_numpy(dtype="int8")
    # Role indicators and all extra columns are derived below, so validation
    # checks only the columns that must actually exist in each input file.
    source_columns = {
        "hours_elapsed",
        "transaction_type",
        "transaction_amount",
        "initiating_customer",
        "initiater_balance_before",
        "recipient_customer",
        "recipient_balance_before",
        "is_flagged_fraud",
    }
    missing_train = source_columns.difference(train.columns)
    missing_test = source_columns.difference(external.columns)
    if missing_train or missing_test:
        raise ValueError(f"Missing train columns: {sorted(missing_train)}; missing test columns: {sorted(missing_test)}")

    print(f"Loaded train={train.shape}, external={external.shape}, positives={int(y.sum())}")
    print("Building strict-prior-hour graph features from training data...")
    train_graph = build_graph_features(train)
    feature_frames = make_feature_frames(train, train_graph)

    # Two nested stratified splits: final holdout is untouched until reporting;
    # calibration chooses the model threshold using development rows only.
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    development_idx, holdout_idx = next(outer.split(np.zeros(len(y)), y))
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    fit_rel, calibration_rel = next(inner.split(np.zeros(len(development_idx)), y[development_idx]))
    fit_idx = development_idx[fit_rel]
    calibration_idx = development_idx[calibration_rel]
    print(f"Split rows: fit={len(fit_idx)}, calibration={len(calibration_idx)}, holdout={len(holdout_idx)}")
    print(f"Split positives: fit={int(y[fit_idx].sum())}, calibration={int(y[calibration_idx].sum())}, holdout={int(y[holdout_idx].sum())}")

    metric_rows: list[dict[str, object]] = []
    holdout_scores: dict[tuple[str, str], np.ndarray] = {}
    best_calibration = None
    best_key = None
    best_cal_y = None
    best_cal_scores = None

    for config, columns in CONFIGS.items():
        X = feature_frames[config]
        positive_weight = float((len(fit_idx) - int(y[fit_idx].sum())) / max(int(y[fit_idx].sum()), 1))
        for algorithm in ALGORITHMS:
            print(f"Training {config} / {algorithm} ({len(columns)} features)...")
            pipeline = make_pipeline(columns, algorithm, positive_weight)
            pipeline.fit(X.iloc[fit_idx], y[fit_idx])
            calibration_scores = pipeline.predict_proba(X.iloc[calibration_idx])[:, 1]
            threshold = best_f1_threshold(y[calibration_idx], calibration_scores)
            calibration_metrics = evaluate_scores(y[calibration_idx], calibration_scores, threshold)
            hold_scores = pipeline.predict_proba(X.iloc[holdout_idx])[:, 1]
            holdout_scores[(config, algorithm)] = hold_scores
            holdout_metrics = evaluate_scores(y[holdout_idx], hold_scores, threshold)
            metric_rows.append({
                "feature_config": config,
                "algorithm": algorithm,
                "n_features": len(columns),
                "fit_rows": len(fit_idx),
                "calibration_rows": len(calibration_idx),
                "holdout_rows": len(holdout_idx),
                "fit_positive": int(y[fit_idx].sum()),
                "calibration_positive": int(y[calibration_idx].sum()),
                "holdout_positive": int(y[holdout_idx].sum()),
                "threshold": threshold,
                "calibration_ap": calibration_metrics["average_precision"],
                "calibration_f1": calibration_metrics["f1"],
                "holdout_ap": holdout_metrics["average_precision"],
                "holdout_roc_auc": holdout_metrics["roc_auc"],
                "holdout_precision": holdout_metrics["precision"],
                "holdout_recall": holdout_metrics["recall"],
                "holdout_f1": holdout_metrics["f1"],
                "tn": holdout_metrics["tn"],
                "fp": holdout_metrics["fp"],
                "fn": holdout_metrics["fn"],
                "tp": holdout_metrics["tp"],
                "alert_count": holdout_metrics["alert_count"],
                "alert_rate": holdout_metrics["alert_rate"],
                "false_alerts_per_100k": holdout_metrics["false_alerts_per_100k"],
            })
            # Prefer the simpler feature set when calibration metrics tie.
            ranking_key = (
                float(calibration_metrics["average_precision"]),
                float(calibration_metrics["f1"]),
                -len(columns),
            )
            if best_calibration is None or ranking_key > best_calibration:
                best_calibration = ranking_key
                best_key = (config, algorithm)
                best_cal_y = y[calibration_idx].copy()
                best_cal_scores = calibration_scores.copy()
            del pipeline, calibration_scores, hold_scores
            gc.collect()

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["calibration_ap", "calibration_f1", "n_features"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    metrics.to_csv(metrics_dir / "model_metrics.csv", index=False)
    metrics.to_parquet(metrics_dir / "model_metrics.parquet", index=False)

    score_table = pd.DataFrame({"row_position": holdout_idx, "is_fraud": y[holdout_idx]})
    for (config, algorithm), scores in holdout_scores.items():
        score_table[f"{config}__{algorithm}"] = scores
    score_table.to_parquet(metrics_dir / "holdout_scores.parquet", index=False)

    selected = metrics.iloc[0]
    selected_config = str(selected["feature_config"])
    selected_algorithm = str(selected["algorithm"])
    selected_threshold = float(selected["threshold"])
    assert best_key == (selected_config, selected_algorithm)
    print(f"Selected by calibration AP/F1: {selected_config} + {selected_algorithm}; threshold={selected_threshold:.6g}")

    plot_model_comparison(metrics, figures_dir / "model_comparison_average_precision.pdf")
    plot_precision_recall_curves(y[holdout_idx], holdout_scores, metrics, figures_dir / "precision_recall_curves.pdf")
    plot_threshold_selection(best_cal_y, best_cal_scores, selected_threshold, figures_dir / "selected_threshold_curve.pdf")

    # Refit the selected pipeline on all labelled training rows, then infer on
    # the untouched external file. No external labels or tuning are used.
    print("Refitting selected model on all training rows...")
    final_positive_weight = float((len(y) - int(y.sum())) / max(int(y.sum()), 1))
    final_pipeline = make_pipeline(CONFIGS[selected_config], selected_algorithm, final_positive_weight)
    final_pipeline.fit(feature_frames[selected_config], y)

    # The selected model currently uses no graph columns.  Avoid reading the
    # external table into a graph-history builder unless a graph configuration
    # is actually selected; this keeps final inference clearly isolated.
    if any(column in CONFIGS[selected_config] for column in GRAPH_COLUMNS):
        print("Building external graph features from train history plus external rows...")
        combined = pd.concat(
            [train.loc[:, GRAPH_INPUT_COLUMNS], external.loc[:, GRAPH_INPUT_COLUMNS]],
            ignore_index=True,
        )
        combined_graph = build_graph_features(combined)
        external_graph = combined_graph.iloc[len(train):].reset_index(drop=True)
    else:
        external_graph = pd.DataFrame(np.nan, index=np.arange(len(external)), columns=GRAPH_COLUMNS)
    external_frames = make_feature_frames(external, external_graph)
    external_scores = final_pipeline.predict_proba(external_frames[selected_config])[:, 1]
    external_output = external.copy()
    external_output["is_fraud_predicted"] = (external_scores >= selected_threshold).astype("int8")
    external_output.index = external_index
    external_path = predictions_dir / "test_fraud_external.parquet"
    external_output.to_parquet(external_path, index=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    external_output.to_parquet(submission_path, index=True)
    pd.DataFrame({"fraud_score": external_scores}).to_parquet(
        predictions_dir / "test_fraud_external_scores.parquet", index=False
    )
    dump(final_pipeline, models_dir / "selected_fraud_pipeline.joblib")
    metadata = {
        "selected_feature_config": selected_config,
        "selected_algorithm": selected_algorithm,
        "threshold": selected_threshold,
        "random_state": RANDOM_STATE,
        "train_rows": len(train),
        "external_rows": len(external),
        "external_positive_predictions": int(external_output["is_fraud_predicted"].sum()),
        "feature_columns": CONFIGS[selected_config],
        "validation": "StratifiedShuffleSplit: 60% fit, 20% calibration, 20% untouched holdout",
    }
    (models_dir / "selected_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved metrics to {metrics_dir}")
    print(f"Saved PDFs to {figures_dir}")
    print(f"Saved model to {models_dir / 'selected_fraud_pipeline.joblib'}")
    print(f"Saved predictions to {external_path}")
    print(f"Saved submission to {submission_path}")
    print(f"External predicted fraud rows: {metadata['external_positive_predictions']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--test", type=Path, default=TEST_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--submission-path", type=Path, default=Path("submission/test_fraud_external.parquet"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
