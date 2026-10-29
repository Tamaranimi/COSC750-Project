"""COSC750 PCA sensitivity: matched five-seed Gaussian Bayes and RBF-SVM.

Run from the existing project directory. Friday is never used for model selection.
The PCA fit is training-only, including within each GroupKFold CV fold.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.ids_stage3.data import load_dataset, split_days, numeric_feature_columns
from src.ids_stage3.models import DiagonalGaussianBayes
from five_seed_sampling import sample_fit_calibration
from temporal_blocks import blocks
from run_feedback_protocol import prior_restore, tune_threshold, rates


def raw(df, features):
    return df[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)


def batch_scores(model, df, features, model_type, batch=20000):
    scores = np.empty(len(df), dtype=np.float64)
    for start in range(0, len(df), batch):
        end = min(start + batch, len(df))
        X = raw(df.iloc[start:end], features)
        if model_type == 'Gaussian':
            # Bypass sklearn Pipeline's fitted-state tag check for the custom Gaussian estimator.
            # All preprocessing steps were fitted on training data by model.fit().
            X_transformed = model[:-1].transform(X)
            p = model.named_steps['gaussian'].predict_proba(X_transformed)[:, 1]
            scores[start:end] = logit(np.clip(p, 1e-8, 1 - 1e-8))
        else:
            scores[start:end] = model.decision_function(X)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--output-dir', default='pca_sensitivity_results')
    ap.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
    ap.add_argument('--components', nargs='+', type=int, default=[10, 20, 40, 60])
    ap.add_argument('--prevalence', type=float, default=0.01)
    ap.add_argument('--models', nargs='+', choices=['Gaussian', 'SVM'], default=['Gaussian', 'SVM'])
    args = ap.parse_args()
    if not 0 < args.prevalence < 1:
        ap.error('prevalence must be between 0 and 1')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full, label = load_dataset(args.data_dir)
    train, val, test = split_days(full)
    features = numeric_feature_columns(full, label)
    del full
    if any(k < 1 or k > len(features) for k in args.components):
        ap.error(f'PCA components must be between 1 and {len(features)}')
    yval = val['__y'].to_numpy(dtype=int)
    ytest = test['__y'].to_numpy(dtype=int)
    rows = []
    for seed in args.seeds:
        fit, cal = sample_fit_calibration(train, seed)
        yfit = fit['__y'].to_numpy(dtype=int)
        ycal = cal['__y'].to_numpy(dtype=int)
        Xfit_raw = raw(fit, features)
        Xcal_raw = raw(cal, features)
        for k in args.components:
            for name in args.models:
                if name == 'Gaussian':
                    model = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=k, svd_solver='full')),
                        ('gaussian', DiagonalGaussianBayes()),
                    ])
                    model.fit(Xfit_raw, yfit)
                    chosen_c = None
                else:
                    pipe = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=k, svd_solver='full')),
                        ('svm', SVC(kernel='rbf', class_weight={0: 1, 1: 100})),
                    ])
                    search = GridSearchCV(pipe, {'svm__C': [1, 10], 'svm__gamma': ['scale']},
                                          scoring='average_precision', cv=GroupKFold(n_splits=3),
                                          n_jobs=1, refit=True)
                    search.fit(Xfit_raw, yfit, groups=blocks(fit))
                    model = search.best_estimator_
                    chosen_c = search.best_params_['svm__C']
                sc = batch_scores(model, cal, features, name)
                sv = batch_scores(model, val, features, name)
                st = batch_scores(model, test, features, name)
                calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(sc.reshape(-1, 1), ycal)
                pv = calibrator.predict_proba(sv.reshape(-1, 1))[:, 1]
                pt = calibrator.predict_proba(st.reshape(-1, 1))[:, 1]
                source_prior = float(ycal.mean())
                pv = prior_restore(pv, source_prior, args.prevalence)
                pt = prior_restore(pt, source_prior, args.prevalence)
                threshold = tune_threshold(yval, pv, args.prevalence, 1, 100)
                fpr, fnr = rates(ytest, pt, threshold)
                rows.append(dict(seed=seed, model=name, pca_components=k,
                                 expected_cost=(1-args.prevalence)*fpr + 100*args.prevalence*fnr,
                                 fpr=fpr, recall=1-fnr, fnr=fnr, threshold=threshold,
                                 source_prevalence=source_prior, svm_best_c=chosen_c,
                                 n_fit=len(fit), n_cal=len(cal)))
                pd.DataFrame(rows).to_csv(out/'pca_sensitivity_partial.csv', index=False)
                print(f'Completed seed {seed}, {name}, PCA-{k}: cost={rows[-1]["expected_cost"]:.5f}', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out/'pca_sensitivity.csv', index=False)
    summary = result.groupby(['model','pca_components'], as_index=False).agg(
        mean_expected_cost=('expected_cost', 'mean'),
        sd_expected_cost=('expected_cost', 'std'),
        mean_fpr=('fpr', 'mean'), mean_recall=('recall', 'mean'), seeds=('seed','nunique'))
    summary.to_csv(out/'pca_summary.csv', index=False)
    (out/'pca_protocol.json').write_text(json.dumps({
        'seeds': args.seeds, 'components': args.components, 'models': args.models,
        'prevalence': args.prevalence, 'c_fp': 1, 'c_fn': 100,
        'fit_calibration': 'disjoint training time blocks from existing sampler',
        'preprocessing': 'median imputation, scaling, PCA fitted on training only; within CV folds for SVM',
        'svm_cv': '3-fold GroupKFold, C in [1,10], gamma scale, average precision',
        'calibration': 'sigmoid on calibration blocks',
        'threshold': 'Thursday validation expected cost', 'test': 'Friday only',
        'uncertainty': 'across-seed sample SD, no cluster bootstrap',
        'comparison_note': 'PCA sensitivity within this experiment; separate original no-PCA experiment for context'
    }, indent=2))
    print(summary.to_string(index=False))
    print('Saved:', out.resolve())

if __name__ == '__main__':
    main()
