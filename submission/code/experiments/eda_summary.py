"""Recreate the principal training-set EDA summaries and report figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PLOT_DIR = Path(__file__).resolve().parents[1] / "plot"
sys.path.insert(0, str(PLOT_DIR))

from data_quality_eda import plot_zero_value_rates  # noqa: E402
from eda_plots import (  # noqa: E402
    get_dataset_quality_metrics,
    plot_label_distribution,
    plot_time_distribution,
    summarize_data_quality,
)
from full_balance_eda import plot_full_balance_rule  # noqa: E402
from transaction_type_eda import plot_transaction_type_fraud_rates  # noqa: E402


def main(train_path: Path, output_dir: Path) -> None:
    data = pd.read_parquet(train_path)
    if "is_fraud" not in data:
        raise ValueError("Training data must contain is_fraud")
    figures = output_dir / "figures" / "eda"
    metrics = output_dir / "metrics"
    figures.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)

    quality = get_dataset_quality_metrics(data)
    quality["fraud_rows"] = int(data["is_fraud"].sum())
    quality["fraud_rate"] = float(data["is_fraud"].mean())
    (metrics / "eda_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    summarize_data_quality(data).to_csv(metrics / "eda_columns.csv")

    plot_label_distribution(data, output_dir=figures)
    plot_time_distribution(data, output_dir=figures)
    plot_zero_value_rates(data, output_dir=figures)
    plot_transaction_type_fraud_rates(data, output_dir=figures)
    plot_full_balance_rule(data, output_dir=figures)
    print(f"Saved EDA summaries to {metrics} and figures to {figures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("train_fraud.parquet"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    main(args.train, args.output)
