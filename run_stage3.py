from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ids_stage3.experiment import run_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Run COSC 750 Stage 3 baseline experiment on corrected CIC-IDS2017.")
    p.add_argument("--data-dir", required=True, help="Directory containing the five corrected CIC-IDS2017 CSV/Parquet day files")
    p.add_argument("--output-dir", default="results", help="Directory for CSV/TeX result files")
    p.add_argument("--prevalence", type=float, default=0.01, help="Validation/test attack prevalence for Stage 3 operating point")
    p.add_argument("--c-fn", type=float, default=100.0, help="False-negative cost; false-positive cost is fixed at 1")
    p.add_argument("--max-train", type=int, default=12000, help="Maximum sampled Mon-Wed training rows per seed")
    p.add_argument("--max-val", type=int, default=10000, help="Maximum sampled Thursday validation rows per seed")
    p.add_argument("--max-test", type=int, default=25000, help="Maximum sampled Friday test rows per seed")
    p.add_argument("--attempted-policy", choices=["benign", "exclude"], default="benign")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_prevalence=args.prevalence,
        c_fn=args.c_fn,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        attempted_policy=args.attempted_policy,
    )
