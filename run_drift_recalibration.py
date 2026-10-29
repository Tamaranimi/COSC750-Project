"""Compare original training-block sigmoid calibration to Thursday-window recalibration.

Both methods use the same frozen Gaussian/SVM model. Thursday is split into an
EARLIER calibration window and LATER threshold-tuning window, by whole 10-minute
blocks. Friday is never used for model fitting, calibration, or threshold tuning.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from src.ids_stage3.data import load_dataset, split_days, numeric_feature_columns
from src.ids_stage3.models import DiagonalGaussianBayes
from temporal_blocks import blocks
from five_seed_sampling import sample_fit_calibration
from svm_pipeline import build_svm_search
from batch_predictions import predict_svm_in_batches
from run_feedback_protocol import prior_restore, tune_threshold, rates

SEEDS=(11,22,33,44,55)

def thursday_windows(df):
    ids=blocks(df)
    unique=np.sort(np.unique(ids))
    # Find a near-half chronological split with both classes on either side.
    candidates=sorted(range(max(2,int(.20*len(unique))), min(len(unique)-2,int(.80*len(unique)))+1),
                      key=lambda k:abs(k-len(unique)/2))
    for k in candidates:
        left=np.isin(ids,unique[:k]); right=~left
        if df.loc[left,'__y'].nunique()==2 and df.loc[right,'__y'].nunique()==2:
            return df.loc[left].copy(),df.loc[right].copy(),str(unique[k-1]),str(unique[k])
    raise ValueError('No chronological Thursday block split with both classes on both sides; do not silently mix times.')

def score_gaussian(model,prep,df,features,batch=20000):
    parts=[]
    for start in range(0,len(df),batch):
        x=df.iloc[start:start+batch][features].replace([np.inf,-np.inf],np.nan).to_numpy(float)
        p=model.predict_proba(prep.transform(x))[:,1]
        parts.append(logit(np.clip(p,1e-8,1-1e-8)))
    return np.concatenate(parts)

def calibrate(scores,y):
    if len(np.unique(y))!=2:raise ValueError('Calibration window lacks both classes')
    return LogisticRegression(C=1e6,max_iter=1000).fit(np.asarray(scores).reshape(-1,1),y)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--output-dir',default='drift_recalibration_results')
    ap.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    ap.add_argument('--prevalence',type=float,default=.01)
    args=ap.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    full,label=load_dataset(args.data_dir)
    train,thursday,friday=split_days(full)
    features=numeric_feature_columns(full,label)
    early,late,last_early,first_late=thursday_windows(thursday)
    print(f'Thursday: earlier calibration={len(early):,} rows; later threshold validation={len(late):,} rows; boundary {last_early} / {first_late}',flush=True)
    y_early=early['__y'].to_numpy(int);y_late=late['__y'].to_numpy(int); y_test=friday['__y'].to_numpy(int)
    rows=[]
    for seed in args.seeds:
        fit,cal=sample_fit_calibration(train,seed)
        y_fit=fit['__y'].to_numpy(int);y_cal=cal['__y'].to_numpy(int)
        prep=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())])
        X_fit=prep.fit_transform(fit[features].replace([np.inf,-np.inf],np.nan).to_numpy(float))
        gb=DiagonalGaussianBayes().fit(X_fit,y_fit)
        svc=build_svm_search()
        svc.fit(fit[features].replace([np.inf,-np.inf],np.nan).to_numpy(float),y_fit,groups=blocks(fit))
        for name,score_fn in [('Gaussian',lambda d:score_gaussian(gb,prep,d,features)),
                              ('SVM',lambda d:predict_svm_in_batches(svc,d,features))]:
            s_cal=score_fn(cal);s_early=score_fn(early);s_late=score_fn(late);s_test=score_fn(friday)
            for strategy,cal_scores,cal_labels in [('training_only',s_cal,y_cal),('thursday_recalibrated',s_early,y_early)]:
                sigmoid=calibrate(cal_scores,cal_labels)
                prior=float(cal_labels.mean())
                # Prior correction assumes class-conditional scores remain stable.
                p_late=prior_restore(sigmoid.predict_proba(s_late.reshape(-1,1))[:,1],prior,args.prevalence)
                p_test=prior_restore(sigmoid.predict_proba(s_test.reshape(-1,1))[:,1],prior,args.prevalence)
                threshold=tune_threshold(y_late,p_late,args.prevalence,1,100)
                fpr,fnr=rates(y_test,p_test,threshold)
                cost=(1-args.prevalence)*fpr+100*args.prevalence*fnr
                # Observed-Friday Brier/log loss: not a deployment-prevalence estimate.
                brier=brier_score_loss(y_test,p_test)
                ll=log_loss(y_test,p_test,labels=[0,1])
                rows.append(dict(seed=seed,model=name,strategy=strategy,train_n=len(fit),
                    calibration_n=len(cal_labels),calibration_attack_rate=prior,
                    thursday_early_n=len(early),thursday_late_n=len(late),
                    prevalence=args.prevalence,threshold=threshold,expected_cost=cost,
                    fpr=fpr,recall=1-fnr,brier_friday_observed_mix=brier,
                    logloss_friday_observed_mix=ll))
                print(f'Completed seed {seed}, {name}, {strategy}: cost={cost:.5f}',flush=True)
        pd.DataFrame(rows).to_csv(out/'recalibration_partial.csv',index=False)
    result=pd.DataFrame(rows)
    result.to_csv(out/'recalibration_sensitivity.csv',index=False)
    summary=result.groupby(['model','strategy'],as_index=False).agg(
        mean_expected_cost=('expected_cost','mean'),sd_expected_cost=('expected_cost','std'),
        mean_fpr=('fpr','mean'),mean_recall=('recall','mean'),
        mean_brier_observed=('brier_friday_observed_mix','mean'),
        mean_logloss_observed=('logloss_friday_observed_mix','mean'),seeds=('seed','nunique'))
    summary.to_csv(out/'recalibration_summary.csv',index=False)
    (out/'recalibration_protocol.json').write_text(json.dumps(dict(
        seeds=args.seeds,prevalence=args.prevalence,c_fp=1,c_fn=100,
        thursday_early_end=last_early,thursday_late_start=first_late,
        model_training='Monday-Wednesday fit blocks',
        comparison='original training-block sigmoid vs new sigmoid on earlier Thursday labelled blocks',
        threshold='later Thursday blocks only; both strategies independently threshold-tuned',
        test='Friday only; no Friday labels used for fitting or tuning',
        caveats=['Thursday early calibration labels are required and are unavailable in true unsupervised deployment',
                 'Brier/log-loss evaluated at observed Friday class mix, unlike 1% hypothetical expected cost',
                 'prior correction assumes stable class-conditional distributions',
                 'This experiment changes calibration data; differences do not isolate temporal drift from sample size/composition',
                 'Five seeds are related samples, not independent datasets']),indent=2))
    print(summary.to_string(index=False));print('Saved:',out.resolve())

if __name__=='__main__':main()
