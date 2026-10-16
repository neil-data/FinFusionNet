"""FinFusionNet CLI: Train and Evaluate LSTM Sequential Experiments (L1 to L6).

Usage:
  python scripts/train_lstm.py --experiment L1 --protocol protocol_b
  python scripts/train_lstm.py --all --protocol protocol_b
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
from pathlib import Path
import pandas as pd

from finfusionnet.evaluation.lstm_pipeline import run_lstm_experiment, LSTM_EXPERIMENT_SPECS


def main():
    parser = argparse.ArgumentParser(description="FinFusionNet LSTM Sequential Experiment Runner")
    parser.add_argument(
        "--experiment",
        type=str,
        default="L1",
        help="Experiment ID: L1, L2, L3, L4, L5, L6, or ALL",
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Maximum epochs per fold",
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

    experiments = list(LSTM_EXPERIMENT_SPECS.keys()) if args.experiment.upper() == "ALL" else [args.experiment.upper()]
    protocols = ["protocol_a", "protocol_b"] if args.protocol == "both" else [args.protocol]

    results = []
    for exp_id in experiments:
        for proto in protocols:
            print(f"\n=======================================================")
            print(f"Running LSTM Experiment {exp_id} under {proto}...")
            print(f"Description: {LSTM_EXPERIMENT_SPECS[exp_id]['description']}")
            print(f"=======================================================")
            res = run_lstm_experiment(
                master_df=master_df,
                experiment_id=exp_id,
                protocol=proto,
                news_days_only=args.news_days_only,
                max_epochs=args.epochs,
            )
            s = res["summary"]
            print(f"Run completed: {res['run_path']}")
            print(f"Date-level Accuracy: {s['primary_metric_equal_weighted_date_accuracy']:.4f}")
            print(f"Pooled Accuracy:     {s['pooled_accuracy']:.4f}")
            print(f"Pooled F1-Score:     {s['pooled_f1']:.4f}")
            print(f"Lift over Always-Up: {s['lift_over_always_up']:+.4f}")
            print(f"Lift over Prev-Dir:  {s['lift_over_previous_direction']:+.4f}")
            results.append(s)

    print("\nAll requested LSTM experiment runs finished successfully.")


if __name__ == "__main__":
    main()
