"""COSC750 training-size sensitivity, matched Gaussian Bayes vs RBF-SVM.

Run from the original project directory. Only fit rows are reduced; calibration,
Thursday validation and Friday test remain fixed within each seed.
"""
from __future__ import annotations
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
from temporal_blocks import blocks
from svm_pipeline import build_svm_search
from batch_predictions import predict_svm_in_batches
from run_feedback_protocol import prior_restore, tune_threshold, rates


def sample_fraction_by_block_and_class(fit, fraction, seed):
    """Nested deterministic row subsets, stratified by temporal block and class.

    Does not create new temporal partitions: all subsets are within original fit
    blocks, disjoint from calibration. Maintains minority examples in each block.
    """
    if fraction == 1:
        return fit.copy()
    group_ids = blocks(fit)
    y = fit['__y'].to_numpy(dtype=int)
    chosen = []
    # Stable per-(block,class) permutations make subsets nested across fractions.
    for group in np.unique(group_ids):
        for cls in (0, 1):
            idx = np.flatnonzero((group_ids == group) & (y == cls))
            if len(idx) == 0:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(np.flatnonzero(np.unique(group_ids) == group)[0]), cls]))
            idx = rng.permutation(idx)
            take = max(1, int(np.floor(len(idx) * fraction)))
            chosen.extend(idx[:take].tolist())
    return fit.iloc[np.sort(chosen)].copy()


def gaussian_scores(model, prep, df, features, batch_size=20000):
    out = np.empty(len(df), dtype=float)
    for start in range(0, len(df), batch_size):
        part = df.iloc[start:start + batch_size]
        x = part[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        p = model.predict_proba(prep.transform(x))[:, 1]
        out[start:start + len(part)] = logit(np.clip(p, 1e-8, 1-1e-8))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--output-dir', default='training_size_results')
    ap.add_argument('--seeds', nargs='+', type=int, default=[11,22,33,44,55])
    ap.add_argument('--fractions', nargs='+', type=float, default=[0.25,0.50,0.75,1.0])
    ap.add_argument('--prevalence', type=float, default=0.01)
    ap.add_argument('--models', nargs='+', choices=['Gaussian','SVM'], default=['Gaussian','SVM'])
    args = ap.parse_args()
    if not 0 < args.prevalence < 1 or any(not 0 < f <= 1 for f in args.fractions):
        ap.error('Prevalence must be in (0,1); fractions must be in (0,1]')
    if len(set(args.fractions)) != len(args.fractions):
        ap.error('Fractions must be unique')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full, label = load_dataset(args.data_dir)
    train, val, test = split_days(full)
    features = numeric_feature_columns(full, label)
    del full
    yv = val['__y'].to_numpy(dtype=int)
    yt = test['__y'].to_numpy(dtype=int)
    rows = []
    for seed in args.seeds:
        fit_full, cal = sample_fit_calibration(train, seed)
        ycal = cal['__y'].to_numpy(dtype=int)
        cal_prior = float(ycal.mean())
        if not 0 < cal_prior < 1:
            raise ValueError(f'Seed {seed}: calibration needs both classes')
        for fraction in sorted(args.fractions):
            fit = sample_fraction_by_block_and_class(fit_full, fraction, seed)
            yf = fit['__y'].to_numpy(dtype=int)
            groups = blocks(fit)
            if len(np.unique(groups)) < 3 or len(np.unique(yf)) != 2:
                raise ValueError(f'Seed {seed}, fraction {fraction}: inadequate classes/blocks')
            xfit = fit[features].replace([np.inf,-np.inf], np.nan).to_numpy(dtype=float)
            for name in args.models:
                best_c = None
                if name == 'Gaussian':
                    prep = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                     ('scale', StandardScaler())])
                    xt = prep.fit_transform(xfit)
                    model = DiagonalGaussianBayes().fit(xt, yf)
                    sc = gaussian_scores(model, prep, cal, features)
                    sv = gaussian_scores(model, prep, val, features)
                    st = gaussian_scores(model, prep, test, features)
                else:
                    search = build_svm_search()
                    search.fit(xfit, yf, groups=groups)
                    best_c = search.best_params_.get('svm__C', search.best_params_.get('svc__C'))
                    sc = predict_svm_in_batches(search, cal, features)
                    sv = predict_svm_in_batches(search, val, features)
                    st = predict_svm_in_batches(search, test, features)
                calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(sc.reshape(-1,1), ycal)
                pv = prior_restore(calibrator.predict_proba(sv.reshape(-1,1))[:,1], cal_prior, args.prevalence)
                pt = prior_restore(calibrator.predict_proba(st.reshape(-1,1))[:,1], cal_prior, args.prevalence)
                threshold = tune_threshold(yv, pv, args.prevalence, 1, 100)
                fpr, fnr = rates(yt, pt, threshold)
                cost = (1-args.prevalence)*fpr + 100*args.prevalence*fnr
                rows.append(dict(seed=seed, model=name, train_fraction=fraction,
                                 n_fit=len(fit), n_fit_attacks=int(yf.sum()),
                                 n_fit_blocks=len(np.unique(groups)), n_cal=len(cal),
                                 expected_cost=cost, fpr=fpr, fnr=fnr, recall=1-fnr,
                                 threshold=threshold, calibration_source_prevalence=cal_prior,
                                 svm_best_c=best_c))
                pd.DataFrame(rows).to_csv(out/'training_size_partial.csv', index=False)
                print(f'Completed seed {seed}, {name}, {fraction:.0%}: n_fit={len(fit)}, cost={cost:.5f}', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out/'training_size_sensitivity.csv', index=False)
    summary = result.groupby(['model','train_fraction'],as_index=False).agg(
        mean_expected_cost=('expected_cost','mean'),
        sd_expected_cost=('expected_cost','std'),
        mean_fpr=('fpr','mean'), mean_recall=('recall','mean'),
        mean_n_fit=('n_fit','mean'), seeds=('seed','nunique'))
    summary.to_csv(out/'training_size_summary.csv', index=False)
    protocol = dict(seeds=args.seeds, fractions=args.fractions, models=args.models,
                    prevalence=args.prevalence, c_fp=1, c_fn=100,
                    training_subset='nested row subsamples within each fit temporal block and class; calibration blocks untouched',
                    fit_calibration='same original disjoint training-time blocks for each seed',
                    preprocessing='median imputation and scaling fitted only on each reduced fit subset; SVM CV preprocessing fit inside folds',
                    svm_cv='existing build_svm_search with 3-fold GroupKFold',
                    calibration='same sigmoid on unchanged calibration blocks',
                    threshold='Thursday expected cost', test='Friday only',
                    uncertainty='sample SD across five seeds; no cluster bootstrap',
                    caution='subsets share underlying fit blocks; seed repeats not independent; row subsampling changes size but not block coverage')
    (out/'training_size_protocol.json').write_text(json.dumps(protocol, indent=2))
    print(summary.to_string(index=False))
    print('Saved:', out.resolve())

if __name__ == '__main__':
    main()
