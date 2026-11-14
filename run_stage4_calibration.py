"""Stage 4 calibration ablation. Raw-score baselines are explicitly heuristic.

Use identical fitted models per seed; compare native Gaussian posterior or
expit(SVM margin) with held-out Platt calibration. Thursday selects thresholds;
Friday is evaluation-only. Probability metrics use observed Friday prevalence,
not a simulated 1% deployment sample.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ids_stage3.data import load_dataset, split_days, numeric_feature_columns
from src.ids_stage3.models import DiagonalGaussianBayes
from five_seed_sampling import sample_fit_calibration
from svm_pipeline import build_svm_search
from batch_predictions import predict_svm_in_batches
from temporal_blocks import blocks
from run_feedback_protocol import prior_restore, rates, cost_at_prior, tune_threshold

SEEDS = (11, 22, 33, 44, 55)


def score_to_uncalibrated_probability(scores, model):
    """Heuristic probability mapping, not a learned or validated calibration."""
    if model not in ('Gaussian', 'SVM'):
        raise ValueError('unknown model')
    return expit(np.asarray(scores, dtype=float))


def model_variants(cal_scores, val_scores, test_scores, ycal, model):
    sigmoid = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.asarray(cal_scores).reshape(-1, 1), ycal)
    return {
        'uncalibrated': (
            score_to_uncalibrated_probability(val_scores, model),
            score_to_uncalibrated_probability(test_scores, model)),
        'platt_sigmoid': (
            sigmoid.predict_proba(np.asarray(val_scores).reshape(-1, 1))[:, 1],
            sigmoid.predict_proba(np.asarray(test_scores).reshape(-1, 1))[:, 1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--output-dir', default='stage4_calibration_results')
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    ap.add_argument('--prevalence', type=float, default=0.01)
    ap.add_argument('--fp-cost', type=float, default=1.0)
    ap.add_argument('--fn-cost', type=float, default=100.0)
    args = ap.parse_args()
    if not 0 < args.prevalence < 1 or min(args.fp_cost, args.fn_cost) <= 0:
        ap.error('prevalence must be between 0 and 1 and costs must be positive')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full, label = load_dataset(args.data_dir)
    train, val, test = split_days(full)
    features = numeric_feature_columns(full, label)
    yv, yt = val['__y'].to_numpy(int), test['__y'].to_numpy(int)
    if len(np.unique(yv)) != 2 or len(np.unique(yt)) != 2:
        raise ValueError('Thursday and Friday must contain both classes')
    rows = []
    for seed in args.seeds:
        fit, cal = sample_fit_calibration(train, seed)
        yfit, ycal = fit['__y'].to_numpy(int), cal['__y'].to_numpy(int)
        source_prior = float(ycal.mean())
        prep = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scaler', StandardScaler())])
        def array(df):
            return df[features].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        xfit = prep.fit_transform(array(fit))
        gb = DiagonalGaussianBayes().fit(xfit, yfit)
        def gb_score(df):
            p = np.clip(gb.predict_proba(prep.transform(array(df)))[:, 1], 1e-8, 1-1e-8)
            return logit(p)
        group = blocks(fit)
        if len(np.unique(group)) < 3:
            raise ValueError('At least three fitting blocks are required')
        svc = build_svm_search()
        svc.fit(array(fit), yfit, groups=group)
        score_sets = {
            'Gaussian': (gb_score(cal), gb_score(val), gb_score(test)),
            'SVM': (predict_svm_in_batches(svc, cal, features),
                    predict_svm_in_batches(svc, val, features),
                    predict_svm_in_batches(svc, test, features)),
        }
        for model, (sc, sv, st) in score_sets.items():
            for variant, (pv, pt) in model_variants(sc, sv, st, ycal, model).items():
                # Prior adjustment is meaningful as a probability correction only
                # when probabilities are calibrated and label-shift holds.
                v_adj = prior_restore(pv, source_prior, args.prevalence)
                t_adj = prior_restore(pt, source_prior, args.prevalence)
                threshold = tune_threshold(yv, v_adj, args.prevalence, args.fp_cost, args.fn_cost)
                fpr, fnr = rates(yt, t_adj, threshold)
                rows.append(dict(seed=seed, model=model, variant=variant,
                                 threshold=threshold, fpr=fpr, fnr=fnr,
                                 expected_cost=cost_at_prior(yt, t_adj, threshold,
                                                             args.prevalence, args.fp_cost, args.fn_cost),
                                 brier_friday_observed_prior=brier_score_loss(yt, pt),
                                 logloss_friday_observed_prior=log_loss(yt, np.clip(pt, 1e-10, 1-1e-10), labels=[0, 1]),
                                 calibration_source_prior=source_prior))
            print(f'Finished seed {seed}, {model}', flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / 'calibration_runs.csv', index=False)
    summary = runs.groupby(['model', 'variant'], as_index=False).agg(
        mean_cost=('expected_cost', 'mean'), sd_cost=('expected_cost', 'std'),
        mean_fpr=('fpr', 'mean'), mean_fnr=('fnr', 'mean'),
        mean_brier=('brier_friday_observed_prior', 'mean'),
        mean_logloss=('logloss_friday_observed_prior', 'mean'),
        n_seeds=('seed', 'nunique'))
    summary.to_csv(out / 'calibration_summary.csv', index=False)
    (out / 'protocol.json').write_text(json.dumps({
        'comparison': 'Platt sigmoid versus heuristic raw-score mapping; same fitted model per seed',
        'uncalibrated_gaussian': 'native Gaussian posterior, clipped to [1e-8, 1-1e-8]',
        'uncalibrated_svm': 'expit(raw decision_function); heuristic only, NOT a calibrated probability',
        'calibration': 'LogisticRegression(C=1e6) fitted on disjoint Monday-Wednesday calibration blocks',
        'model_training': 'original cost-weighted SVM and Gaussian; unchanged Stage 3 settings',
        'threshold': 'Thursday validation expected-cost minimization independently per variant',
        'prior_adjustment': 'odds correction applied for consistency; probabilistic interpretation of uncalibrated arm is NOT justified',
        'probability_metrics': 'Friday observed prevalence; NOT target 1% prevalence; exploratory temporal-generalization metrics',
        'Friday_policy': 'Friday labels only for evaluation, not fitting, calibration, or threshold tuning',
        'seeds': args.seeds, 'prevalence': args.prevalence,
        'false_positive_cost': args.fp_cost, 'false_negative_cost': args.fn_cost,
        'data_directory': args.data_dir,
    }, indent=2))
    print(summary.to_string(index=False))
    print('Saved:', out.resolve())

if __name__ == '__main__':
    main()
