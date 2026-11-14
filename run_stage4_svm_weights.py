"""Stage 4 controlled SVM class-weight ablation (no Friday tuning)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ids_stage3.data import load_dataset, split_days, numeric_feature_columns
from five_seed_sampling import sample_fit_calibration
from svm_pipeline import build_svm_search
from batch_predictions import predict_svm_in_batches
from temporal_blocks import blocks
from run_feedback_protocol import prior_restore, cost_at_prior, rates, tune_threshold

SEEDS = (11, 22, 33, 44, 55)
WEIGHTS = {'weighted_1_100': {0: 1, 1: 100}, 'unweighted': None}


def configured_search(class_weight):
    """Reuse original fold-safe pipeline and search, varying only class_weight."""
    search = build_svm_search()
    search.estimator.set_params(svm__class_weight=class_weight)
    return search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--output-dir', default='stage4_svm_weight_results')
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    ap.add_argument('--prevalence', type=float, default=0.01)
    ap.add_argument('--fp-cost', type=float, default=1.0)
    ap.add_argument('--fn-cost', type=float, default=100.0)
    args = ap.parse_args()
    if not 0 < args.prevalence < 1 or min(args.fp_cost, args.fn_cost) <= 0:
        ap.error('prevalence must be in (0,1); costs must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    full, _ = load_dataset(args.data_dir)
    train, validation, test = split_days(full)
    features = numeric_feature_columns(full, _)
    yv = validation['__y'].to_numpy(int)
    yt = test['__y'].to_numpy(int)
    if len(np.unique(yv)) != 2 or len(np.unique(yt)) != 2:
        raise ValueError('Thursday and Friday must both contain two classes')
    rows = []
    for seed in args.seeds:
        fit, cal = sample_fit_calibration(train, seed)
        yfit = fit['__y'].to_numpy(int)
        ycal = cal['__y'].to_numpy(int)
        groups = blocks(fit)
        if len(np.unique(groups)) < 3:
            raise ValueError('Need at least three training blocks')
        Xfit = fit[features].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        for variant, class_weight in WEIGHTS.items():
            search = configured_search(class_weight)
            search.fit(Xfit, yfit, groups=groups)
            scores_cal = predict_svm_in_batches(search, cal, features)
            scores_val = predict_svm_in_batches(search, validation, features)
            scores_test = predict_svm_in_batches(search, test, features)
            sigmoid = LogisticRegression(C=1e6, max_iter=1000).fit(
                np.asarray(scores_cal).reshape(-1, 1), ycal)
            source_prior = float(ycal.mean())
            pv = prior_restore(sigmoid.predict_proba(
                np.asarray(scores_val).reshape(-1, 1))[:, 1], source_prior, args.prevalence)
            pt = prior_restore(sigmoid.predict_proba(
                np.asarray(scores_test).reshape(-1, 1))[:, 1], source_prior, args.prevalence)
            threshold = tune_threshold(yv, pv, args.prevalence, args.fp_cost, args.fn_cost)
            fpr, fnr = rates(yt, pt, threshold)
            rows.append(dict(seed=seed, variant=variant, threshold=threshold,
                             fpr=fpr, fnr=fnr,
                             expected_cost=cost_at_prior(yt, pt, threshold, args.prevalence,
                                                         args.fp_cost, args.fn_cost),
                             best_C=search.best_params_['svm__C'],
                             best_gamma=search.best_params_['svm__gamma'],
                             fit_rows=len(fit), calibration_rows=len(cal),
                             validation_rows=len(validation), test_rows=len(test)))
            print(f'Finished seed {seed}, {variant}', flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / 'svm_weight_runs.csv', index=False)
    summary = results.groupby('variant', as_index=False).agg(
        mean_cost=('expected_cost', 'mean'), sd_cost=('expected_cost', 'std'),
        mean_fpr=('fpr', 'mean'), mean_fnr=('fnr', 'mean'), n_seeds=('seed', 'nunique'))
    summary.to_csv(output / 'svm_weight_summary.csv', index=False)
    (output / 'protocol.json').write_text(json.dumps({
        'comparison': 'SVM class_weight only; independent fitting and calibration per variant',
        'weights': WEIGHTS, 'seeds': args.seeds, 'prevalence': args.prevalence,
        'false_positive_cost': args.fp_cost, 'false_negative_cost': args.fn_cost,
        'same_data_splits': 'Monday-Wednesday fit/calibration; Thursday validation; Friday test',
        'cv': 'GroupKFold(3), average_precision; same C and gamma grid',
        'calibration': 'Platt sigmoid on disjoint training calibration blocks',
        'prior_adjustment': 'source calibration prevalence to target prevalence',
        'threshold': 'Thursday validation expected-cost minimization',
        'test_policy': 'Friday labels used only for final evaluation',
        'data_directory': args.data_dir,
    }, indent=2))
    print(summary.to_string(index=False))
    print('Saved:', output.resolve())


if __name__ == '__main__':
    main()
