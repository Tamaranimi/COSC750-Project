"""Exploratory known-family holdout, separate from the archived Friday protocol.

Whole 10-minute blocks are withheld from Monday-Wednesday before model fitting.
Both methods use the same calibration and Thursday threshold selection.
The held-out training-day blocks are NOT a chronological future test, and their
scores must NOT be pooled with Friday unseen-family scores as a fair comparison.
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
from temporal_blocks import blocks
from five_seed_sampling import sample_fit_calibration
from svm_pipeline import build_svm_search
from batch_predictions import predict_svm_in_batches
from run_feedback_protocol import prior_restore, tune_threshold, family_report


def select_holdout_blocks(train, label, seed, fraction=0.2):
    """Sample whole groups while preserving at least one group for each family in fitting."""
    group_ids = blocks(train)
    unique = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    families = train[label].astype(str).str.strip().to_numpy()
    attack = train['__y'].to_numpy() == 1
    group_families = {g: set(families[(group_ids == g) & attack]) for g in unique}
    counts = {fam: sum(fam in v for v in group_families.values()) for fam in set(families[attack])}
    # Greedily select a varied subset; never remove the final fitting block of a family.
    selected = []
    remaining = dict(counts)
    for g in rng.permutation(unique):
        if len(selected) >= max(1, int(round(len(unique) * fraction))):
            break
        if any(remaining[fam] <= 1 for fam in group_families[g]):
            continue
        selected.append(g)
        for fam in group_families[g]:
            remaining[fam] -= 1
    if not selected:
        raise ValueError('No eligible holdout blocks; inspect per-family temporal coverage')
    held = train.loc[np.isin(group_ids, selected)].copy()
    fitting = train.loc[~np.isin(group_ids, selected)].copy()
    if fitting['__y'].nunique() != 2 or held['__y'].nunique() != 2:
        raise ValueError('Whole-block split lacks benign or attack class; choose another seed')
    if set(blocks(held)).intersection(set(blocks(fitting))):
        raise AssertionError('Block leakage detected')
    return fitting, held


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--output-dir', default='verification_known_family')
    ap.add_argument('--seeds', type=int, nargs='+', default=[11,22,33,44,55])
    ap.add_argument('--prevalence', type=float, default=0.01)
    ap.add_argument('--holdout-fraction', type=float, default=0.2)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full, label = load_dataset(args.data_dir)
    train, validation, friday = split_days(full)
    features = numeric_feature_columns(full, label)
    rows = []
    metadata = []
    for seed in args.seeds:
        remaining, known_holdout = select_holdout_blocks(train, label, seed, args.holdout_fraction)
        fit, calibration = sample_fit_calibration(remaining, seed)
        yfit = fit['__y'].to_numpy(int)
        ycal = calibration['__y'].to_numpy(int)
        yv = validation['__y'].to_numpy(int)
        if len(np.unique(yv)) != 2:
            raise ValueError('Thursday validation lacks one class')
        prep = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scale', StandardScaler())])
        def raw(df):
            return df[features].replace([np.inf,-np.inf], np.nan).to_numpy(float)
        xf = prep.fit_transform(raw(fit))
        xc = prep.transform(raw(calibration))
        xv = prep.transform(raw(validation))
        xh = prep.transform(raw(known_holdout))
        gb = DiagonalGaussianBayes().fit(xf, yfit)
        def gscore(x):
            return logit(np.clip(gb.predict_proba(x)[:,1], 1e-8, 1-1e-8))
        svc = build_svm_search()
        svc.fit(raw(fit), yfit, groups=blocks(fit))
        score_sets = {
            'Gaussian': (gscore(xc), gscore(xv), gscore(xh)),
            'SVM': (predict_svm_in_batches(svc, calibration, features),
                    predict_svm_in_batches(svc, validation, features),
                    predict_svm_in_batches(svc, known_holdout, features)),
        }
        known = set(fit.loc[fit['__y']==1,label].astype(str).str.strip())
        holdout_families = known_holdout[label].astype(str).str.strip().to_numpy()
        yh = known_holdout['__y'].to_numpy(int)
        for method, (sc, sv, sh) in score_sets.items():
            calibrator = LogisticRegression(C=1e6,max_iter=1000).fit(sc.reshape(-1,1), ycal)
            source_prior = float(ycal.mean())
            v = prior_restore(calibrator.predict_proba(sv.reshape(-1,1))[:,1], source_prior, args.prevalence)
            h = prior_restore(calibrator.predict_proba(sh.reshape(-1,1))[:,1], source_prior, args.prevalence)
            threshold = tune_threshold(yv, v, args.prevalence, 1, 100)
            for rec in family_report(yh, holdout_families, h, threshold, args.prevalence, 1, 100, known):
                if rec['group'] != 'known':
                    continue
                rows.append(dict(seed=seed, method=method, prevalence=args.prevalence,
                                 threshold=threshold, evaluation='held-out Monday-Wednesday blocks', **rec))
        metadata.append(dict(seed=seed, fit_rows=len(fit), calibration_rows=len(calibration),
                             known_holdout_rows=len(known_holdout), holdout_attack_families=sorted(set(holdout_families[yh==1]))))
        print(f'Completed known-family seed {seed}', flush=True)
    if not rows:
        raise ValueError('No known attack-family results generated')
    df = pd.DataFrame(rows)
    df.to_csv(out/'known_family_holdout.csv', index=False)
    summary = df.groupby(['method','family'], as_index=False).agg(
        seeds=('seed','nunique'), attack_recall_mean=('attack_recall','mean'),
        attack_recall_std=('attack_recall','std'), n_attacks_per_seed_mean=('n_attacks','mean'))
    summary.to_csv(out/'known_family_summary.csv', index=False)
    (out/'protocol.json').write_text(json.dumps(dict(
        purpose='Separate exploratory known-family diagnostic; NOT Friday chronological test',
        seeds=args.seeds, prevalence=args.prevalence, holdout_fraction=args.holdout_fraction,
        block_split='whole Monday-Wednesday 10-minute blocks excluded before fitting',
        calibration='disjoint remaining Monday-Wednesday training blocks, sigmoid for both methods',
        threshold='Thursday expected-cost validation, no known-holdout tuning',
        limitations=['known holdout is within training weekdays, not a future-time test',
                     'different time periods and attack families prevent controlled known-versus-unseen causal attribution',
                     'not directly comparable with previously archived Friday model scores',
                     'family recall is conditional on family; family-conditional cost uses shared benign FPR'],
        per_seed=metadata), indent=2))
    print(summary.to_string(index=False))
    print('Saved',out.resolve())

if __name__ == '__main__':
    main()
