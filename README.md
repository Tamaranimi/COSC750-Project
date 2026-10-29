# COSC 750 Stage 3 — Cost-Sensitive Bayesian IDS



**Researcher:** Nimi Kenanagha

**Institution:** Bowie State University

**Course:** COSC 750 — Pattern Recognition I

**Semester:** Fall 2026



## Project Overview



This repository contains the Python implementation and experimental results for the research project:



**Cost-Sensitive Bayesian Decisions for Network Intrusion Detection: Performance Under Class Imbalance and Temporal Drift**



The project investigates whether Gaussian Bayesian intrusion detection can reduce expected misclassification cost compared with a calibrated, cost-weighted RBF Support Vector Machine (SVM).



The study evaluates class imbalance, temporal drift, Gaussian mixture complexity, PCA dimensionality, training-set size, and posterior recalibration.



## Dataset



**Dataset:** CIC-IDS2017

**Source:** Canadian Institute for Cybersecurity (CIC)

**Official website:** https://cicresearch.ca/CICDataset/CIC-IDS-2017/browse.php?p=CIC-IDS-2017%2FCSVs



The experimental dataset was prepared from the original CIC-IDS2017 CSV release and stored as eight Parquet files.



Recorded processing counts:



- 2,830,743 records loaded after initial preparation.

- 203 exact duplicate records removed during experimental processing.

- 2,830,540 records remaining.

- 80 numeric features used.



The local dataset directory is `CICIDS2017_Timestamped_Clean`.



The dataset is not redistributed through this repository. Users must obtain it from the official source.



**Provenance limitation:** The exact CSV-to-Parquet preprocessing procedure remains to be verified.



## Numerical Preprocessing

The baseline experiment uses the following preprocessing procedure:

- Select numeric network-flow features, excluding identifiers, timestamps, labels, and internal metadata.
- Replace positive and negative infinity with missing values (`NaN`).
- Impute missing values using the median of each feature.
- Standardize features using `StandardScaler`.
- Fit the imputer and scaler on training data only.
- Apply the fitted transformations to validation and test data without refitting.

This training-only fitting procedure prevents validation and test data from influencing the baseline preprocessing parameters.

## Experimental Protocol



- **Training:** Monday–Wednesday

- **Validation:** Thursday

- **Testing:** Friday



The matched-protocol experiment uses separate timestamp blocks for fitting, calibration, and validation.



Five random seeds: 11, 22, 33, 44, 55.



Primary operating conditions:



- Attack prevalence: 1%

- False-positive cost: 1

- False-negative cost: 100

- Primary metric: expected misclassification cost



Secondary measures include recall, false-positive rate, calibration measures, and attack-family detection performance.



## Models



**Gaussian Bayes:** Diagonal Gaussian generative classifier with Bayesian minimum-risk decisions.



**RBF-SVM:** Cost-weighted SVM with sigmoid probability calibration and validation-based threshold selection.



**Gaussian Mixture Models:** Sensitivity analysis with 1, 2, and 4 mixture components.



**Constant-prediction baselines:** Always-pass and always-alert decisions.



## Completed Experiments



### Experiment 1 — Matched Gaussian Bayes vs. RBF-SVM



Script: `run_feedback_protocol.py`



Results directory: `feedback_results_all5/`



| Model | Mean expected cost |

|---|---|

| Gaussian Bayes | 0.9561 ± 0.0734 |

| RBF-SVM | 0.5989 ± 0.2496 |



The RBF-SVM achieved lower mean expected cost.



### Experiment 2 — Gaussian Mixture Sensitivity



Script: `run_gmm_sensitivity.py`



Results directory: `gmm_sensitivity_results/`



| Components | Mean expected cost |

|---|---|

| 1 | 0.9760 |

| 2 | 1.0298 |

| 4 | 1.0011 |



Increasing mixture complexity did not consistently improve expected cost.



### Experiment 3 — PCA Sensitivity



Script: `run_pca_sensitivity.py`



Results directory: `pca_sensitivity_results/`



PCA dimensions: 10, 20, 40, and 60.



The lowest observed Gaussian mean cost was at 40 dimensions; the lowest SVM mean cost was at 60 dimensions.



### Experiment 4 — Training-Size Sensitivity



Script: `run_training_size_sensitivity.py`



Results directory: `training_size_results/`



Training fractions: 25%, 50%, 75%, and 100%.



### Experiment 5 — Temporal Recalibration



Script: `run_drift_recalibration.py`



Results directory: `drift_recalibration_results/`



| Model | Training-only calibration | Thursday recalibration |

|---|---|---|

| Gaussian Bayes | 0.9567 | 0.9731 |

| RBF-SVM | 0.5863 | 0.5863 |



Thursday recalibration did not improve mean expected decision cost.



## Software Environment



- Python 3.13.14

- NumPy 2.5.3

- pandas 3.0.6

- scikit-learn 1.9.1

- SciPy 1.18.1

- PyArrow 25.0.1



Installed dependencies are recorded in `requirements_verified.txt`.



Install using:



```powershell

python -m pip install -r requirements_verified.txt

```



## Running the Experiments



Run the tests:



```powershell

python -m pytest -q

```



Previously verified local result: **6 tests passed in 1.42 seconds.**



Matched-protocol experiment:



```powershell

python run_feedback_protocol.py --data-dir "C:\Users\kenan\Downloads\CICIDS2017_Timestamped_Clean" --output-dir feedback_results_all5

```



The supported command-line options for this script and the four sensitivity scripts must be verified against their source code before publication.



## Reproducibility and Limitations



- Friday test data are held out from fitting and threshold selection.

- Results represent five related random-seed runs, not five independent datasets.

- Deployment prevalence is evaluated through prevalence-aware expected-cost calculations.

- Prior correction assumes sufficiently stable class-conditional distributions.

- Thursday recalibration requires labeled data.

- Recalibration changes both calibration timing and sample composition.

- CIC-IDS2017 is a historical controlled-environment dataset.



## Key Findings



Under the primary experimental conditions, the calibrated RBF-SVM achieved lower mean expected cost than Gaussian Bayes.



Increasing Gaussian mixture complexity did not produce consistent improvements, and temporal recalibration did not improve mean decision cost.



These findings emphasize model specification, calibration, prevalence, temporal evaluation, and explicit misclassification costs.



## Repository Status



All five experiment result sets are available. The local automated test suite previously passed.



Independent reproduction from the original CSV files remains subject to verification of the preprocessing pipeline and experiment execution instructions.




