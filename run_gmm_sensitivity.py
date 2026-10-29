"""GMM sensitivity extension of the completed COSC750 matched protocol.

Does not overwrite feedback_results_all5. No Friday labels used for fitting,
calibration, hyperparameter selection, or threshold tuning.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from src.ids_stage3.data import load_dataset, split_days, numeric_feature_columns
from five_seed_sampling import sample_fit_calibration
from run_feedback_protocol import prior_restore, tune_threshold, rates


def score_llr(g0, g1, X, batch_size=25000):
    result=np.empty(len(X),dtype=np.float64)
    for a in range(0,len(X),batch_size):
        b=min(a+batch_size,len(X))
        result[a:b]=g1.score_samples(X[a:b])-g0.score_samples(X[a:b])
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--output-dir',default='gmm_sensitivity_results')
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    ap.add_argument('--components',nargs='+',type=int,default=[1,2,4])
    ap.add_argument('--prevalence',type=float,default=0.01)
    ap.add_argument('--max-iter',type=int,default=100)
    args=ap.parse_args()
    if not 0<args.prevalence<1: raise ValueError('Prevalence must be in (0,1)')
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    full,label=load_dataset(args.data_dir)
    train,val,test=split_days(full)
    features=numeric_feature_columns(full,label)
    del full
    rows=[]
    for seed in args.seeds:
        fit,cal=sample_fit_calibration(train,seed)
        yfit=fit['__y'].to_numpy(dtype=int)
        ycal=cal['__y'].to_numpy(dtype=int)
        yval=val['__y'].to_numpy(dtype=int)
        ytest=test['__y'].to_numpy(dtype=int)
        imp=SimpleImputer(strategy='median')
        scale=StandardScaler()
        def raw(df):
            return df[features].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=np.float64)
        Xfit=scale.fit_transform(imp.fit_transform(raw(fit)))
        Xcal=scale.transform(imp.transform(raw(cal)))
        Xval=scale.transform(imp.transform(raw(val)))
        Xtest=scale.transform(imp.transform(raw(test)))
        for k in args.components:
            if k<1 or min(np.bincount(yfit))<k: raise ValueError('Invalid component count')
            g0=GaussianMixture(n_components=k,covariance_type='diag',reg_covar=1e-6,
                max_iter=args.max_iter,random_state=seed,n_init=1).fit(Xfit[yfit==0])
            g1=GaussianMixture(n_components=k,covariance_type='diag',reg_covar=1e-6,
                max_iter=args.max_iter,random_state=seed,n_init=1).fit(Xfit[yfit==1])
            sc=score_llr(g0,g1,Xcal)
            sv=score_llr(g0,g1,Xval)
            st=score_llr(g0,g1,Xtest)
            sigmoid=LogisticRegression(C=1e6,max_iter=1000).fit(sc.reshape(-1,1),ycal)
            pv=sigmoid.predict_proba(sv.reshape(-1,1))[:,1]
            pt=sigmoid.predict_proba(st.reshape(-1,1))[:,1]
            cal_prior=float(ycal.mean())
            vadj=prior_restore(pv,cal_prior,args.prevalence)
            tadj=prior_restore(pt,cal_prior,args.prevalence)
            threshold=tune_threshold(yval,vadj,args.prevalence,1,100)
            fpr,fnr=rates(ytest,tadj,threshold)
            rows.append(dict(seed=seed,method=f'GMM-{k}',components=k,prevalence=args.prevalence,
                expected_cost=(1-args.prevalence)*fpr+100*args.prevalence*fnr,
                fpr=fpr,fnr=fnr,recall=1-fnr,threshold=threshold,
                brier_test_observed_prevalence=brier_score_loss(ytest,pt),
                log_loss_test_observed_prevalence=log_loss(ytest,pt,labels=[0,1]),
                converged_benign=g0.converged_,converged_attack=g1.converged_,
                n_iter_benign=g0.n_iter_,n_iter_attack=g1.n_iter_,
                fit_records=len(fit),calibration_records=len(cal),
                validation_records=len(val),test_records=len(test)))
            print(f'Completed seed {seed}, GMM-{k}: cost={rows[-1]["expected_cost"]:.5f}',flush=True)
        pd.DataFrame(rows).to_csv(out/'gmm_sensitivity_partial.csv',index=False)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'gmm_sensitivity.csv',index=False)
    summary=frame.groupby('components').agg(
        mean_expected_cost=('expected_cost','mean'),sd_expected_cost=('expected_cost','std'),
        mean_fpr=('fpr','mean'),mean_recall=('recall','mean'),
        seeds=('seed','nunique')).reset_index()
    summary.to_csv(out/'gmm_summary.csv',index=False)
    (out/'gmm_protocol.json').write_text(json.dumps(dict(seeds=args.seeds,components=args.components,
        prevalence=args.prevalence,c_fp=1,c_fn=100,
        fit_calibration='same disjoint training-time blocks as matched protocol',
        calibration='sigmoid on calibration blocks',
        threshold='Thursday expected cost',test='Friday only',
        uncertainty='across-seed sample SD; no cluster bootstrap in this extension',
        note='GMM uses diagonal covariance and per-class EM fitting; GMM-1 is not guaranteed numerically identical to earlier Gaussian implementation'),indent=2))
    print('\n',summary.to_string(index=False),'\nSaved:',out.resolve())

if __name__=='__main__': main()
