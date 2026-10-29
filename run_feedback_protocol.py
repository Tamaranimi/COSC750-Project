"""Supplemental professor-feedback experiment; does NOT overwrite archived Stage 3 results.

Usage: python run_feedback_protocol.py --data-dir PATH --output-dir feedback_results
Requires timestamps and attack-family labels in the CIC-IDS2017 weekday files.
No Friday labels are used for model selection.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss, log_loss
from src.ids_stage3.data import load_dataset, split_days, numeric_feature_columns
from src.ids_stage3.models import DiagonalGaussianBayes
from temporal_blocks import blocks
from five_seed_sampling import sample_fit_calibration
from svm_pipeline import build_svm_search
from batch_predictions import predict_svm_in_batches
from fast_bootstrap import bootstrap_blocks_fast

PREVALENCES=(0.001,0.01,0.05,0.10,0.16)
SEEDS=(11,22,33,44,55)

def group_cap(df, cap, seed):
    """Select entire temporal blocks only; never cut a block to meet cap."""
    if len(df)<=cap: return df.copy()
    rng=np.random.default_rng(seed)
    labels=blocks(df)
    uniques=np.unique(labels)
    rng.shuffle(uniques)
    chosen=[]; total=0
    for group in uniques:
        count=int(np.sum(labels==group))
        if total+count>cap: continue
        chosen.append(group); total+=count
    if total<100: raise ValueError('Not enough whole blocks under cap; increase cap')
    return df.loc[np.isin(labels,chosen)].copy()

def split_fit_calibration(df, seed, fraction=0.25):
    rng=np.random.default_rng(seed)
    ids=np.unique(blocks(df)); rng.shuffle(ids)
    n=max(1,int(round(len(ids)*fraction)))
    cal=df.loc[np.isin(blocks(df),ids[:n])].copy()
    fit=df.loc[~np.isin(blocks(df),ids[:n])].copy()
    for name,part in [('fit',fit),('calibration',cal)]:
        if part['__y'].nunique()!=2: raise ValueError(f'{name} partition has only one class; adjust group split')
    return fit,cal

def prior_restore(prob,source_prior,target_prior):
    """Correct calibrated posterior odds under label/prior shift only."""
    if not (0<source_prior<1 and 0<target_prior<1):raise ValueError('priors must be between 0 and 1')
    p=np.clip(np.asarray(prob,dtype=float),1e-10,1-1e-10)
    return expit(logit(p)+logit(target_prior)-logit(source_prior))

def rates(y,score,threshold):
    y=np.asarray(y); pred=np.asarray(score)>=threshold
    if not np.any(y==0) or not np.any(y==1): return np.nan,np.nan
    return float(np.mean(pred[y==0])),float(np.mean(~pred[y==1]))

def cost_at_prior(y,score,threshold,prior,c_fp,c_fn):
    fpr,fnr=rates(y,score,threshold)
    return float((1-prior)*c_fp*fpr+prior*c_fn*fnr)

def tune_threshold(y,score,prior,c_fp,c_fn):
    if len(np.unique(y))!=2: raise ValueError('Validation requires both classes')
    q=np.unique(np.r_[np.quantile(score,np.linspace(0,1,301)),c_fp/(c_fp+c_fn),0,1])
    costs=np.array([cost_at_prior(y,score,t,prior,c_fp,c_fn) for t in q])
    return float(q[int(np.argmin(costs))])

def family_report(y,families,score,threshold,prior,c_fp,c_fn,known):
    benign=(y==0); fpr=float(np.mean(score[benign]>=threshold))
    result=[]
    for fam in sorted(set(families[y==1])):
        mask=(y==1)&(families==fam)
        fnr=float(np.mean(score[mask]<threshold))
        result.append(dict(family=fam,group='known' if fam in known else 'unseen',n_attacks=int(mask.sum()),attack_recall=1-fnr,benign_fpr=fpr,conditional_expected_cost=(1-prior)*c_fp*fpr+prior*c_fn*fnr))
    return result

def bootstrap_blocks(y,score,threshold,prior,c_fp,c_fn,group_ids,seed=123,n_boot=200):
    """Cluster bootstrap: resample whole Friday time blocks, not independent flows."""
    rng=np.random.default_rng(seed); uniq=np.unique(group_ids)
    indices={g:np.flatnonzero(group_ids==g) for g in uniq}
    vals=[]
    for _ in range(n_boot):
        draw=rng.choice(uniq,size=len(uniq),replace=True)
        ix=np.concatenate([indices[g] for g in draw]); yy=y[ix]
        if len(np.unique(yy))!=2:continue
        vals.append(cost_at_prior(yy,score[ix],threshold,prior,c_fp,c_fn))
    if len(vals)<20:return (np.nan,np.nan,len(vals))
    return (*map(float,np.quantile(vals,[0.025,0.975])),len(vals))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--output-dir',default='feedback_results')
    ap.add_argument('--max-train',type=int,default=12000);ap.add_argument('--max-validation',type=int,default=10000)
    ap.add_argument('--max-test',type=int,default=25000);ap.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    full,label=load_dataset(args.data_dir)
    train,val,test=split_days(full)
    features=numeric_feature_columns(full,label)
    known=set(train.loc[train['__y']==1,label].astype(str).str.strip())
    all_rows=[];fam_rows=[]
    for seed in args.seeds:
        fit,cal=sample_fit_calibration(train,seed)
        va=val
        te=test
        yfit=fit["__y"].to_numpy(int); ycal=cal["__y"].to_numpy(int); yv=va["__y"].to_numpy(int)
        # Transform Friday with exactly the same fit-only imputer/scaler: rebuild transform on fit data.
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        prep=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())])
        Xfit=prep.fit_transform(fit[features].replace([np.inf,-np.inf],np.nan).to_numpy(float))
        Xcal=prep.transform(cal[features].replace([np.inf,-np.inf],np.nan).to_numpy(float))
        Xv=prep.transform(va[features].replace([np.inf,-np.inf],np.nan).to_numpy(float))
        Xt=prep.transform(te[features].replace([np.inf,-np.inf],np.nan).to_numpy(float));yt=te['__y'].to_numpy(int)
        if len(np.unique(yv))!=2 or len(np.unique(yt))!=2:raise ValueError('Validation/test missing a class')
        gb=DiagonalGaussianBayes().fit(Xfit,yfit)
        gb_cal=logit(np.clip(gb.predict_proba(Xcal)[:,1],1e-8,1-1e-8))
        gb_val=logit(np.clip(gb.predict_proba(Xv)[:,1],1e-8,1-1e-8))
        gb_test=logit(np.clip(gb.predict_proba(Xt)[:,1],1e-8,1-1e-8))
        group=blocks(fit)
        if len(np.unique(group))<3:raise ValueError('Need >=3 training blocks for group CV')
        svc=build_svm_search()
        svc.fit(fit[features].replace([np.inf,-np.inf],np.nan).to_numpy(float),yfit,groups=group)
        scores={'Gaussian':(gb_cal,gb_val,gb_test),'SVM':(
            predict_svm_in_batches(svc,cal,features),
            predict_svm_in_batches(svc,va,features),
            predict_svm_in_batches(svc,te,features)
        )}
        for name,(sc,sv,st) in scores.items():
            # Same Platt sigmoid calibrator, same held-out training blocks, for BOTH models.
            sigmoid=LogisticRegression(C=1e6,max_iter=1000).fit(sc.reshape(-1,1),ycal)
            pv=sigmoid.predict_proba(sv.reshape(-1,1))[:,1]
            pt=sigmoid.predict_proba(st.reshape(-1,1))[:,1]
            pcal=sigmoid.predict_proba(sc.reshape(-1,1))[:,1]
            cal_prior=float(ycal.mean())
            # Probability quality at source calibration prevalence; no test-driven tuning.
            brier=float(brier_score_loss(yt,pt)); ll=float(log_loss(yt,pt,labels=[0,1]))
            for pi in PREVALENCES:
                v_adj=prior_restore(pv,cal_prior,pi)
                t_adj=prior_restore(pt,cal_prior,pi)
                threshold=tune_threshold(yv,v_adj,pi,1,100)
                fpr,fnr=rates(yt,t_adj,threshold)
                low,high,nb=bootstrap_blocks_fast(yt,t_adj,threshold,pi,1,100,blocks(te),seed=seed)
                all_rows.append(dict(seed=seed,method=name,prevalence=pi,threshold=threshold,expected_cost=(1-pi)*fpr+100*pi*fnr,fpr=fpr,fnr=fnr,brier_test_observed_prevalence=brier,log_loss_test_observed_prevalence=ll,cluster_ci_low=low,cluster_ci_high=high,cluster_bootstrap_replicates=nb,calibration_source_prevalence=cal_prior))
                families=te[label].astype(str).str.strip().to_numpy()
                for rec in family_report(yt,families,t_adj,threshold,pi,1,100,known):
                    fam_rows.append(dict(seed=seed,method=name,prevalence=pi,**rec))
        print(f'Completed seed {seed}',flush=True)
    pd.DataFrame(all_rows).to_csv(out/'fair_comparison.csv',index=False)
    pd.DataFrame(fam_rows).to_csv(out/'attack_families.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(seeds=args.seeds,prevalences=PREVALENCES,grouping='whole 10-minute timestamp blocks',calibration='identical sigmoid on disjoint training blocks',threshold='identical validation expected cost',uncertainty='whole Friday 10-minute-block bootstrap',prior_shift='posterior odds correction assumes invariant class-conditional feature distributions',limitations='family-conditional expected cost uses common benign FPR and attack-family-specific FNR; not a separate family prevalence estimate'),indent=2))
    print('Saved',out.resolve())
if __name__=='__main__':main()

