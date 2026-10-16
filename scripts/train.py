"""FinFusionNet CLI: Train and Evaluate XGBoost Experiments (X1 to X6).

Usage:
  python scripts/train.py --experiment X1 --protocol protocol_b
  python scripts/train.py --all --protocol protocol_b
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import os
from pathlib import Path
import pandas as pd

from finfusionnet.evaluation.pipeline import run_experiment, EXPERIMENT_SPECS


def main():
    parser = argparse.ArgumentParser(description="FinFusionNet XGBoost Experiment Runner")
    parser.add_argument(
        "--experiment",
        type=str,
        default="X1",
        help="Experiment ID: X1, X2, X3, X4, X5, X6, or ALL",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="protocol_b",
        choices=["protocol_a", "protocol_b", "both"],
        help="Evaluation protocol",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/master_dataset/master_ticker_day.parquet",
        help="Path to precomputed master dataset parquet",
    )
    parser.add_argument(
        "--news-days-only",
        action="store_true",
        help="Run secondary evaluation strictly on news days (has_news == 1)",
    )
    args = parser.parse_args()

    data_path = Path(args.dataset)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {data_path}. Please build it first or specify --dataset."
        )

    print(f"Loading master dataset from {data_path}...")
    master_df = pd.read_parquet(data_path)
    print(f"Loaded {len(master_df)} rows across {master_df['ticker'].nunique()} tickers.")

    experiments = list(EXPERIMENT_SPECS.keys()) if args.experiment.upper() == "ALL" else [args.experiment.upper()]
    protocols = ["protocol_a", "protocol_b"] if args.protocol == "both" else [args.protocol]

    results = []
    for exp_id in experiments:
        for proto in protocols:
            print(f"\n=======================================================")
            print(f"Running Experiment {exp_id} under {proto}...")
            print(f"Description: {EXPERIMENT_SPECS[exp_id]['description']}")
            print(f"=======================================================")
            res = run_experiment(
                master_df=master_df,
                experiment_id=exp_id,
                protocol=proto,
                news_days_only=args.news_days_only,
            )
            s = res["summary"]
            print(f"Run completed: {res['run_path']}")
            print(f"Date-level Accuracy: {s['primary_metric_equal_weighted_date_accuracy']:.4f}")
            print(f"Pooled Accuracy:     {s['pooled_accuracy']:.4f}")
            print(f"Pooled F1-Score:     {s['pooled_f1']:.4f}")
            print(f"Lift over Always-Up: {s['lift_over_always_up']:+.4f}")
            print(f"Lift over Prev-Dir:  {s['lift_over_previous_direction']:+.4f}")
            results.append(s)

    print("\nAll requested experiment runs finished successfully.")


if __name__ == "__main__":
    main()
