"""One command: tests, matched protocol, known-family diagnostic, and paper-value check.

Requires local CICIDS2017_Timestamped_Clean inputs. Does not redistribute data.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

EXPECTED = {(0.001,'Gaussian'):0.1374,(0.001,'SVM'):0.0918,
            (0.01,'Gaussian'):0.9561,(0.01,'SVM'):0.5989,
            (0.05,'Gaussian'):2.8700,(0.05,'SVM'):2.1826,
            (0.1,'Gaussian'):5.2868,(0.1,'SVM'):3.2931,
            (0.16,'Gaussian'):0.8400,(0.16,'SVM'):3.5758}
RECALL = {('Gaussian','Bot'):0.0,('Gaussian','DDoS'):21.03,
          ('Gaussian','PortScan'):2.48,('SVM','Bot'):21.89,
          ('SVM','DDoS'):23.27,('SVM','PortScan'):69.28}

def run(*args):
    print('RUN:', ' '.join(map(str,args)),flush=True)
    subprocess.run([sys.executable,*map(str,args)],check=True)

def check(out):
    fair=pd.read_csv(out/'matched'/'fair_comparison.csv')
    families=pd.read_csv(out/'matched'/'attack_families.csv')
    if set(fair.seed) != {11,22,33,44,55}:
        raise AssertionError('Expected five seeds')
    for (pi,method),expected in EXPECTED.items():
        got=fair[(fair.prevalence.round(6)==pi)&(fair.method==method)].expected_cost.mean()
        if abs(got-expected)>0.000051:
            raise AssertionError(f'{method} prevalence {pi}: got {got:.6f}, expected {expected:.4f}')
    for (method,family),expected in RECALL.items():
        d=families[(families.prevalence.round(6)==0.01)&(families.method==method)&(families.family==family)]
        got=d.attack_recall.mean()*100
        if abs(got-expected)>0.0051:
            raise AssertionError(f'{method} {family}: got {got:.4f}%, expected {expected:.2f}%')
    print('PASS: matched-protocol paper costs and six unseen-family recalls reproduce at reported precision')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--output-dir',default='verification_complete')
    ap.add_argument('--skip-known',action='store_true',help='Skip additional exploratory known-family model fitting')
    args=ap.parse_args()
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    run('-m','pytest','-q')
    run('run_feedback_protocol.py','--data-dir',args.data_dir,'--output-dir',out/'matched')
    check(out)
    if not args.skip_known:
        run('run_known_family_holdout.py','--data-dir',args.data_dir,'--output-dir',out/'known_family')
    print('Verification complete:',out.resolve())

if __name__=='__main__':main()
