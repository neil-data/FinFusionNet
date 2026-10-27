"""FinFusionNet: Master Orchestrator for Experiment 4 (FinBERT + Fusion).

Executes the complete Experiment 4 workflow sequentially:
1. Resumes and completes FinBERT sentiment feature extraction from headline cache.
2. Generates daily ticker-day sentiment features parquet.
3. Executes Protocol B 45-fold walk-forward evaluation for XGBoost, LR, and LSTM.
4. Runs paired date-block bootstrap (10,000 samples) and McNemar tests with Holm-Bonferroni correction.
5. Performs tripartite comparisons against F0 price-only and Exp3B VADER, plus subgroup analyses.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.process_gnews_all50_finbert import process_all50_gnews_and_extract_finbert
from scripts.run_experiment_4_finbert import run_experiment_4
from scripts.compute_experiment_4_comparison import analyze_experiment_4


def main():
    t_start = time.time()
    print("=" * 80)
    print("STARTING COMPLETE EXPERIMENT 4 (FINBERT + FUSION) PIPELINE")
    print("=" * 80, flush=True)

    # Step 1: Feature Extraction
    print("\n>>> STEP 1: FinBERT Headline Scoring & Daily Feature Aggregation <<<", flush=True)
    process_all50_gnews_and_extract_finbert()

    # Step 2: Protocol B Walk-Forward Evaluation
    print("\n>>> STEP 2: Protocol B 45-Fold Walk-Forward Model Training & Testing <<<", flush=True)
    run_experiment_4()

    # Step 3: Tripartite Comparison & Subgroup Analysis
    print("\n>>> STEP 3: Tripartite Comparisons & Subgroup Breakdown <<<", flush=True)
    analyze_experiment_4()

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 4 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed / 60:.1f} MINUTES")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
