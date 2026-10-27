"""FinFusionNet: Master Orchestrator for Experiment 5 (Temporal Cross-Attention Fusion).

Executes the complete Experiment 5 pipeline:
1. Validates prerequisites (master dataset, Parkinson volatility target, All-50 FinBERT daily features).
2. Runs Protocol B 45-fold walk-forward cross-attention training with dual heads.
3. Consolidates predictions on the exact common evaluation cohort (N=45,136).
4. Computes comprehensive classification and volatility regression metrics.
5. Executes Diebold-Mariano tests, paired date-block bootstrap (10,000 resamples), McNemar tests, and Holm-Bonferroni correction.
6. Generates subgroup analysis and sector breakdowns.
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment_5_cross_attention import run_experiment_5
from scripts.compute_experiment_5_comparison import run_experiment_5_analysis


def main():
    t0 = time.time()
    print("=" * 80)
    print("FINFUSIONNET EXPERIMENT 5: TEMPORAL CROSS-ATTENTION FUSION PIPELINE")
    print("=" * 80, flush=True)

    print("\n>>> STEP 1: Protocol B 45-Fold Walk-Forward Training & Inference <<<", flush=True)
    run_experiment_5(max_epochs=5, batch_size=512)

    print("\n>>> STEP 2: Comprehensive Statistical Evaluation & Benchmarking <<<", flush=True)
    run_experiment_5_analysis()

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 5 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed / 60:.1f} MINUTES")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
