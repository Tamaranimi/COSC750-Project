# COSC 750 Stage 3 — Cost-Sensitive Bayesian Intrusion Detection

**Researcher:** Nimi Kenanagha  
**Institution:** Bowie State University  
**Course:** COSC 750 — Pattern Recognition I  
**Semester:** Fall 2026

## 1. Project Overview

**Research Title:** Cost-Sensitive Bayesian Decisions for Network Intrusion Detection: Performance Under Class Imbalance and Temporal Drift

This research investigates the performance of Gaussian Bayesian classification and a cost-weighted Radial Basis Function Support Vector Machine (RBF-SVM) for network intrusion detection under class imbalance, asymmetric misclassification costs, and temporal changes in attack traffic.

The principal objective is to determine whether Bayesian minimum-risk decisions can reduce expected misclassification cost compared with a calibrated SVM under a matched experimental protocol.

The study includes:

- Gaussian Bayesian classification and cost-weighted RBF-SVM comparison.
- Probability calibration and expected-cost threshold selection.
- Evaluation under multiple assumed attack prevalences.
- Temporal training, validation, and testing.
- Known-family and previously unseen attack-family diagnostics.
- Gaussian mixture complexity analysis.
- Principal Component Analysis (PCA) sensitivity.
- Training-size sensitivity.
- Temporal recalibration analysis.
- Reproducibility checks across five random seeds.

## 2. Dataset

**Dataset:** CIC-IDS2017  
**Source:** Canadian Institute for Cybersecurity  
**Official dataset:** https://cicresearch.ca/CICDataset/CIC-IDS-2017/browse.php?p=CIC-IDS-2017%2FCSVs

The matched-protocol experiments use a locally prepared dataset named `CICIDS2017_Timestamped_Clean`, stored as eight Parquet files.

| Dataset characteristic | Recorded value |
|---|---:|
| Records loaded | 2,830,743 |
| Exact duplicate records removed | 203 |
| Records remaining | 2,830,540 |
| Numeric features | 80 |

The dataset is not redistributed through this repository.

**Data provenance limitation:** The complete original CSV-to-Parquet preparation procedure has not yet been independently verified. The locally prepared dataset is distinct from the separately archived Distrinet baseline dataset and should not be treated as identical.

## 3. Research Methodology

### Temporal evaluation

The primary chronological experiment uses:

- **Monday–Wednesday:** Training and calibration.
- **Thursday:** Validation and decision-threshold selection.
- **Friday:** Held-out evaluation.

The matched-protocol experiment separates model fitting and probability calibration using disjoint timestamp blocks.

**Random seeds:** 11, 22, 33, 44, and 55.

### Models

**Gaussian Bayes:** A diagonal Gaussian generative classifier using Bayesian minimum-risk decisions.

**RBF-SVM:** A cost-weighted Support Vector Machine with sigmoid probability calibration and validation-based threshold selection.

**Additional experiments:** Gaussian mixture models, PCA dimensionality reduction, training-size sensitivity, and recalibration.

### Cost-sensitive evaluation

The primary operating assumptions are:

| Parameter | Value |
|---|---:|
| False-positive cost | 1 |
| False-negative cost | 100 |
| Primary assumed attack prevalence | 1% |
| Primary metric | Expected misclassification cost |

Expected cost is calculated as:

\[
R=(1-\pi)C_{FP}FPR+\pi C_{FN}FNR
\]

Where:

- \(\pi\) is the assumed attack prevalence.
- \(C_{FP}\) is the false-positive cost.
- \(C_{FN}\) is the false-negative cost.
- \(FPR\) is the false-positive rate.
- \(FNR\) is the false-negative rate.

The experiment also examines prevalence assumptions of 0.1%, 1%, 5%, 10%, and 16%.

## 4. Matched-Protocol Experimental Results

The following results are means across five seeds. The ± values describe variability across those seeds.

| Attack prevalence | Gaussian Bayes expected cost | RBF-SVM expected cost | Best constant-decision cost |
|---|---:|---:|---:|
| 0.1% | 0.1374 ± 0.0349 | 0.0918 ± 0.0200 | 0.1000 |
| 1% | 0.9561 ± 0.0734 | 0.5989 ± 0.2496 | 0.9900 |
| 5% | 2.8700 ± 1.7540 | 2.1826 ± 0.8046 | 0.9500 |
| 10% | 5.2868 ± 4.0069 | 3.2931 ± 1.5950 | 0.9000 |
| 16% | 0.8400 ± 0.0000 | 3.5758 ± 2.4188 | 0.8400 |

### Interpretation

At the primary 1% attack prevalence, the RBF-SVM achieved approximately **37.4% lower mean expected cost** than Gaussian Bayes.

However, at 5% and 10% prevalence, both learned models performed worse than the best constant-decision baseline.

At 16% prevalence, the Gaussian model's mean cost of 0.8400 corresponds to always predicting attack.

These results demonstrate that lower expected cost at one prevalence does not guarantee superior performance across all deployment conditions.

**Results:** `matched/fair_comparison.csv`

## 5. Previously Unseen Attack-Family Evaluation

The Friday test set contains attack families absent from the Monday–Wednesday training period:

- Bot
- DDoS
- PortScan

The following table reports mean attack recall across five seeds at 1% assumed prevalence.

| Unseen attack family | Attack records | Gaussian Bayes recall | RBF-SVM recall |
|---|---:|---:|---:|
| Bot | 1,966 | 0.00% | 21.89% |
| DDoS | 128,027 | 21.03% | 23.27% |
| PortScan | 158,930 | 2.48% | 69.28% |

The RBF-SVM achieved higher mean recall for all three unseen attack families under the primary operating condition.

Nevertheless, the relatively low recall for some attack families indicates that detecting previously unseen attack behavior remains difficult.

**Results:** `matched/attack_families.csv`

## 6. Known Attack-Family Holdout Experiment

An additional exploratory experiment evaluates attack families represented during training.

**Script:** `run_known_family_holdout.py`

**Results directory:** `known family/`

### Experimental procedure

- Whole 10-minute timestamp blocks are withheld from Monday–Wednesday data before model fitting.
- Remaining training blocks are used for model fitting and separate probability calibration.
- Thursday data are used for expected-cost threshold selection.
- Withheld Monday–Wednesday blocks are evaluated independently.
- Five random seeds are used.

### Known-family results

Mean attack recall at 1% assumed prevalence:

| Known attack family | Eligible seeds | Gaussian Bayes recall | RBF-SVM recall |
|---|---:|---:|---:|
| DoS slowloris | 1 | 100.00% | 94.21% |
| FTP-Patator | 1 | 100.00% | 100.00% |
| SSH-Patator | 3 | 82.50% | 33.33% |

**Interpretation:** The known-family diagnostic shows that model performance varies substantially across attack families.

The DoS slowloris and FTP-Patator summaries each contain only one eligible seed. SSH-Patator contains three eligible seeds and substantial variability.

**Important limitation:** This known-family holdout is not a future-time Friday test. Its scores must not be directly compared with Friday unseen-family scores to claim a causal effect of attack novelty. The evaluation periods and family compositions differ.

**Result files:**

- `known family/known_family_holdout.csv`
- `known family/known_family_summary.csv`
- `known family/protocol.json`

## 7. Additional Sensitivity Experiments

### Gaussian mixture complexity

**Script:** `run_gmm_sensitivity.py`

| Mixture components | Mean expected cost |
|---|---:|
| 1 | 0.9760 |
| 2 | 1.0298 |
| 4 | 1.0011 |

Increasing mixture complexity did not consistently reduce expected cost.

**Results:** `gmm_sensitivity_results/`

### PCA dimensionality sensitivity

**Script:** `run_pca_sensitivity.py`

| PCA dimensions | Gaussian Bayes cost | RBF-SVM cost |
|---|---:|---:|
| 10 | 1.0564 | 0.8124 |
| 20 | 0.9414 | 0.7037 |
| 40 | 0.8959 | 0.6005 |
| 60 | 0.9832 | 0.5989 |

The lowest observed Gaussian cost occurred at 40 dimensions, while the lowest SVM cost occurred at 60 dimensions.

**Results:** `pca_sensitivity_results/`

### Training-size sensitivity

**Script:** `run_training_size_sensitivity.py`

| Training fraction | Gaussian Bayes cost | RBF-SVM cost |
|---|---:|---:|
| 25% | 0.9638 | 0.6720 |
| 50% | 0.9594 | 0.6733 |
| 75% | 0.9606 | 0.6119 |
| 100% | 0.9561 | 0.5989 |

Increasing training size produced modest changes in Gaussian performance and a larger overall improvement in SVM performance.

**Results:** `training_size_results/`

### Temporal recalibration

**Script:** `run_drift_recalibration.py`

| Model | Training-only calibration | Thursday recalibration |
|---|---:|---:|
| Gaussian Bayes | 0.9567 | 0.9731 |
| RBF-SVM | 0.5863 | 0.5863 |

Thursday recalibration did not improve mean expected cost in this experiment.

**Results:** `drift_recalibration_results/`

## 8. Repository Organization

```text
COSC750-Project/
├── README.md
├── requirements.txt
├── requirements_verified.txt
├── RUN_STAGE3_WINDOWS.bat
├── run_stage3.py
├── run_feedback_protocol.py
├── run_known_family_holdout.py
├── run_gmm_sensitivity.py
├── run_pca_sensitivity.py
├── run_training_size_sensitivity.py
├── run_drift_recalibration.py
├── verify_stage3.py
├── batch_predictions.py
├── src/
│   └── ids_stage3/
├── tests/
├── results/
├── feedback_results_all5/
├── known family/
│   ├── known_family_holdout.csv
│   ├── known_family_summary.csv
│   └── protocol.json
├── matched/
│   ├── attack_families.csv
│   ├── fair_comparison.csv
│   └── protocol.json
├── gmm_sensitivity_results/
├── pca_sensitivity_results/
├── training_size_results/
└── drift_recalibration_results/
```

The folder names shown reflect the current GitHub repository.

## 9. Software Environment

The recorded local experimental environment includes:

- Python 3.13.14
- NumPy 2.5.3
- pandas 3.0.6
- scikit-learn 1.9.1
- SciPy 1.18.1
- PyArrow 25.0.1

Install the recorded dependencies:

```powershell
python -m pip install -r requirements_verified.txt
```

## 10. Reproducing the Experiments

Clone the repository:

```powershell
git clone https://github.com/Tamaranimi/COSC750-Project.git
cd COSC750-Project
```

Install dependencies:

```powershell
python -m pip install -r requirements_verified.txt
```

Run the automated tests:

```powershell
python -m pytest -q
```

The automated test suite previously passed locally with six tests.

### Matched-protocol experiment

```powershell
python run_feedback_protocol.py --data-dir "C:\path\to\CICIDS2017_Timestamped_Clean" --output-dir matched
```

### Known-family experiment

```powershell
python run_known_family_holdout.py --data-dir "C:\path\to\CICIDS2017_Timestamped_Clean" --output-dir "known family"
```

### Combined verification

```powershell
python verify_stage3.py --data-dir "C:\path\to\CICIDS2017_Timestamped_Clean" --output-dir verification_complete
```

The combined verification script is designed to run the tests, reproduce the matched-protocol experiment, check reported numerical results, and run the known-family diagnostic.

The matched-protocol numerical checks previously passed locally at the reported precision, and the known-family experiment completed for all five seeds.

**Current GitHub dependency gap:** Before a fresh clone can run the newer experiment scripts, the helper modules `temporal_blocks.py`, `five_seed_sampling.py`, and `svm_pipeline.py` must be added to the repository or their imports must be resolved. Successful execution on the original local machine does not establish that a fresh GitHub clone is fully reproducible.

## 11. Research Limitations

1. CIC-IDS2017 represents historical traffic from a controlled experimental environment.
2. The exact local CSV-to-Parquet preparation process requires further documentation.
3. Five seeds provide repeated measurements but do not constitute five independent datasets.
4. Deployment-prevalence adjustment assumes sufficiently stable class-conditional feature distributions.
5. Calibration and threshold selection may be sensitive to temporal distribution changes.
6. Known-family and unseen-family evaluations use different holdout protocols and cannot support a direct causal novelty comparison.
7. Some known attack families have results for fewer than five eligible seeds.
8. Strong performance under one prevalence or cost assumption does not guarantee strong performance under another.
9. The original archived baseline and the later matched-protocol extension use distinct dataset preparations and should be interpreted separately.
10. Complete fresh-clone reproducibility requires resolving the missing helper-module dependencies.

## 12. Key Findings and Conclusion

The matched-protocol results show that the calibrated, cost-weighted RBF-SVM achieved lower mean expected misclassification cost than Gaussian Bayes at the primary 1% attack prevalence.

However, both models experienced substantial performance limitations under other prevalence assumptions. The Friday attack-family evaluation also demonstrated difficulty detecting previously unseen attacks, particularly for the Gaussian classifier.

The additional known-family experiment provides useful diagnostic evidence but does not establish that attack novelty alone explains the performance differences.

Overall, the research highlights the importance of calibrated probabilities, asymmetric error costs, temporal validation, prevalence sensitivity, appropriate baseline comparisons, and transparent reproducibility in network intrusion detection.

## 13. Repository and Research Status

**Completed:**

- Baseline and matched-protocol model comparisons.
- Five-seed matched-protocol evaluation.
- Attack-prevalence sensitivity.
- Friday unseen-family evaluation.
- Exploratory known-family holdout.
- Gaussian mixture sensitivity.
- PCA sensitivity.
- Training-size sensitivity.
- Temporal recalibration.
- Local matched-protocol verification against reported values.

**Remaining reproducibility work:**

- Upload missing helper modules required by the new experiment scripts.
- Document the complete dataset preparation pipeline.
- Verify the full workflow from a clean GitHub clone.

---

**Project:** COSC 750 — Pattern Recognition I  
**Institution:** Bowie State University  
**Researcher:** Nimi Kenanagha  
**Academic term:** Fall 2026
