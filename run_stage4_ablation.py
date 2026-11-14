"""Stage 4 threshold ablation: same fitted models, only decision rule changes.

Runs on prepared CICIDS2017 weekday Parquet files. No Friday labels are used
for model fitting, calibration, or threshold selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ids_stage3.data import load_dataset, split_days, numeric_feature_columns
from src.ids_stage3.models import DiagonalGaussianBayes
from five_seed_sampling import sample_fit_calibration
from svm_pipeline import build_svm_search
from batch_predictions import predict_svm_in_batches
from temporal_blocks import blocks
from run_feedback_protocol import prior_restore, cost_at_prior, tune_threshold

SEEDS = (11, 22, 33, 44, 55)


def threshold_variants(y_val, p_val, prevalence, fp_cost, fn_cost):
    """Same calibrated probabilities, varying only the decision rule."""
    return {
        "fixed_0_5": 0.5,
        "bayes_cost_threshold": fp_cost / (fp_cost + fn_cost),
        "thursday_optimized": tune_threshold(y_val, p_val, prevalence, fp_cost, fn_cost),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='stage4_ablation_results')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--prevalence', type=float, default=0.01)
    parser.add_argument('--fp-cost', type=float, default=1.0)
    parser.add_argument('--fn-cost', type=float, default=100.0)
    args = parser.parse_args()
    if not 0 < args.prevalence < 1 or min(args.fp_cost, args.fn_cost) <= 0:
        parser.error('prevalence must be in (0,1) and costs must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    full, label = load_dataset(args.data_dir)
    train, validation, test = split_days(full)
    features = numeric_feature_columns(full, label)
    yv = validation['__y'].to_numpy(dtype=int)
    yt = test['__y'].to_numpy(dtype=int)
    rows = []
    for seed in args.seeds:
        fit, cal = sample_fit_calibration(train, seed)
        yfit = fit['__y'].to_numpy(dtype=int)
        ycal = cal['__y'].to_numpy(dtype=int)
        preprocess = Pipeline([('imputer', SimpleImputer(strategy='median')),
                               ('scale', StandardScaler())])
        def numeric(df):
            return df[features].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        Xfit = preprocess.fit_transform(numeric(fit))
        gaussian = DiagonalGaussianBayes().fit(Xfit, yfit)
        def gaussian_scores(df):
            probs = gaussian.predict_proba(preprocess.transform(numeric(df)))[:, 1]
            return logit(np.clip(probs, 1e-8, 1 - 1e-8))
        svm = build_svm_search()
        svm.fit(numeric(fit), yfit, groups=blocks(fit))
        models = {
            'Gaussian': (gaussian_scores(cal), gaussian_scores(validation), gaussian_scores(test)),
            'SVM': (predict_svm_in_batches(svm, cal, features),
                    predict_svm_in_batches(svm, validation, features),
                    predict_svm_in_batches(svm, test, features)),
        }
        for model, (cal_scores, val_scores, test_scores) in models.items():
            sigmoid = LogisticRegression(C=1e6, max_iter=1000)
            sigmoid.fit(np.asarray(cal_scores).reshape(-1, 1), ycal)
            source_prior = float(ycal.mean())
            pv = prior_restore(sigmoid.predict_proba(np.asarray(val_scores).reshape(-1, 1))[:, 1],
                               source_prior, args.prevalence)
            pt = prior_restore(sigmoid.predict_proba(np.asarray(test_scores).reshape(-1, 1))[:, 1],
                               source_prior, args.prevalence)
            for variant, threshold in threshold_variants(yv, pv, args.prevalence,
                                                          args.fp_cost, args.fn_cost).items():
                predicted = pt >= threshold
                fpr = float(predicted[yt == 0].mean())
                fnr = float((~predicted[yt == 1]).mean())
                rows.append(dict(seed=seed, model=model, variant=variant,
                                 threshold=float(threshold), fpr=fpr, fnr=fnr,
                                 expected_cost=cost_at_prior(yt, pt, threshold,
                                                             args.prevalence, args.fp_cost, args.fn_cost),
                                 fit_rows=len(fit), calibration_rows=len(cal),
                                 validation_rows=len(validation), test_rows=len(test)))
        print(f'Finished seed {seed}', flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / 'ablation_runs.csv', index=False)
    summary = results.groupby(['model', 'variant'], as_index=False).agg(
        mean_cost=('expected_cost', 'mean'), sd_cost=('expected_cost', 'std'),
        mean_fpr=('fpr', 'mean'), mean_fnr=('fnr', 'mean'), n_seeds=('seed', 'nunique'))
    summary.to_csv(output / 'ablation_summary.csv', index=False)
    (output / 'protocol.json').write_text(json.dumps({
        'seeds': args.seeds, 'prevalence': args.prevalence,
        'false_positive_cost': args.fp_cost, 'false_negative_cost': args.fn_cost,
        'data_directory': str(args.data_dir),
        'comparison': 'Decision threshold only; identical fitted and calibrated models within each seed',
        'thresholds': ['fixed_0_5', 'bayes_cost_threshold', 'thursday_optimized'],
        'validation_policy': 'Thursday selects tuned threshold; Friday evaluation only',
        'interpretation_note': 'This is a threshold ablation, not a calibration or SVM class-weight ablation.'
    }, indent=2))
    print(summary.to_string(index=False))
    print('Saved:', output.resolve())


if __name__ == '__main__':
    main()
