# COSC 750 Stage 3 - Cost-Sensitive Bayesian IDS

Nimi Kenanagha, Bowie State University, Fall 2026.

This repository reproduces the Stage 3 baseline table for the project **Cost-Sensitive Bayesian Decisions for Network Intrusion Detection: Performance Under Class Imbalance and Temporal Drift**.

## What the experiment does

- Loads the corrected/Distrinet CIC-IDS2017 day files from a local directory.
- Removes exact duplicate rows **before** the temporal split.
- Uses a chronological split: **Monday-Wednesday train, Thursday validation, Friday test**.
- Keeps Friday untouched for final evaluation; hyperparameters use 3-fold CV on training only, and the SVM threshold is chosen on Thursday only.
- Repeats the operating-point experiment for seeds `11, 22, 33, 44, 55` and reports mean +/- standard deviation.
- Stage 3 operating point: **1% attack prevalence**, `c_FP=1`, `c_FN=100`.
- Required baselines:
  - **B1:** own NumPy diagonal Gaussian Bayes with MAP threshold 0.5.
  - **B2:** Cost-weighted RBF SVM (scikit-learn), used as a strong published discriminative baseline for direct comparison with the Bayesian approach.
  - **B3:** always-pass trivial floor.
- Also reports a **preliminary proposed-method** result: the same Gaussian posterior with the minimum-risk threshold `1/101`.

## Important data assumption

The script expects the same corrected CIC-IDS2017 release used in Stages 1-2. It recursively discovers `.csv`, `.parquet`, or `.pq` files. Filenames should contain weekdays (Monday-Friday). If exactly five files have no weekday names, the loader maps sorted file order to Monday-Friday and prints a warning; verify that mapping before using the results.

The project treats labels containing `Attempted` as benign by default, consistent with the Stage 1/2 protocol. Use `--attempted-policy exclude` for the planned sensitivity check.


## Setup - Windows

Download the corrected CIC-IDS2017 dataset separately.

Open PowerShell in the project folder and run:

```powershell
.\RUN_STAGE3_WINDOWS.bat "PATH_TO_DATASET"
```

Replace PATH_TO_DATASET with the directory containing
the five weekday dataset files or their parent directory.

The script installs the required dependencies and
reproduces the Stage 3 results.

Run the automated tests with:

```powershell
python -m pytest -q
```


## Output files

After a successful run:

- `results/baseline_runs.csv` - one row per method and seed.
- `results/baseline_summary.csv` - mean +/- standard deviation summary.
- `results/baseline_table_rows.tex` - LaTeX rows for the paper.
- `results/baseline_table_beamer_rows.tex` - LaTeX rows for the slides.
- `results/run_metadata.json` - split sizes, package versions, feature list, and operating-point settings.

## Reproduce the baseline table with one command

Windows:

```bat
.\RUN_STAGE3_WINDOWS.bat "PATH_TO_DATASET"
```

Linux/macOS:

```bash
./run_stage3.sh
```

## Tests

```bash
python -m pytest -q
```

## Fair-comparison rules fixed before the test result is observed

1. Deduplicate before splitting.
2. Train on Monday-Wednesday, tune threshold on Thursday, test on Friday.
3. Five fixed seeds.
4. Same sampled rows and numeric features for every model within a seed.
5. Primary metric is expected cost; recall, FPR, macro-F1, and PR-AUC are secondary.
6. SVM hyperparameters use training-only cross-validation; the test set is never used for model or threshold selection.
7. Stage 3 primary operating point is 1% attack prevalence with `c_FN=100`; broader prevalence/cost/PCA sensitivity is reserved for Stage 4.

## Reproducibility note

The dataset is intentionally not redistributed in this repository. Its license and source is documented in the paper.
