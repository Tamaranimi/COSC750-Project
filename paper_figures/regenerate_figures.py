"""Regenerate all three vector PDF figures used in the Stage 4 paper.

Uses the exact saved per-seed expected costs and paired statistics from
GitHub tag stage4-final-2026. No model training or CIC-IDS2017 data needed.

Run: python regenerate_figures.py
Dependencies: numpy, pandas, matplotlib
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

threshold = pd.read_csv(ROOT / 'figure_data_threshold.csv')
weights = pd.read_csv(ROOT / 'figure_data_weights.csv')
paired = pd.read_csv(ROOT / 'figure_data_paired.csv')
assert len(threshold) == 30 and len(weights) == 10 and len(paired) == 5
assert set(threshold.seed) == {11,22,33,44,55}
assert set(weights.seed) == {11,22,33,44,55}
# Reconcile against committed five-seed summary tables, 6 decimals.
assert abs(threshold.query("model=='SVM' and variant=='thursday_optimized'").expected_cost.mean()-0.5989240867012431) < 1e-12
assert abs(threshold.query("model=='Gaussian' and variant=='thursday_optimized'").expected_cost.mean()-0.9561255447379728) < 1e-12
assert abs(weights.query("variant=='unweighted'").expected_cost.mean()-0.5605313386197699) < 1e-12

plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
styles = [('fixed_0_5','Fixed 0.5'),('bayes_cost_threshold','Bayesian cost threshold'),('thursday_optimized','Thursday optimized')]
fig, axs = plt.subplots(1,2,figsize=(10.7,3.6),sharey=True,layout='constrained')
for ax,model in zip(axs,['Gaussian','SVM']):
    for variant,label in styles:
        d=threshold[(threshold.model==model)&(threshold.variant==variant)].sort_values('seed')
        ax.plot(d.seed,d.expected_cost,'-o',label=label,markersize=4,linewidth=1.4)
    ax.set_title(model)
    ax.set_xticks([11,22,33,44,55]); ax.set_xlabel('Matched seed')
    ax.grid(alpha=.22)
    ax.legend(fontsize=7,loc='best')
axs[0].set_ylabel('Friday expected cost (lower is better)')
fig.savefig(OUT/'threshold_ablation.pdf',bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(9.6,3.9),layout='constrained')
y=np.arange(len(paired))
ax.errorbar(paired.mean_difference,y,xerr=np.array([paired.mean_difference-paired.ci_low,paired.ci_high-paired.mean_difference]),fmt='o',capsize=3,color='#17639a',ecolor='#17639a')
ax.axvline(0,ls='--',lw=1,color='#555555')
ax.set_yticks(y,paired.comparison); ax.invert_yaxis()
ax.set_xlabel('Paired mean cost difference (first minus second), 95% t CI')
ax.set_title('Exploratory ablation effects across five matched seeds')
ax.grid(axis='x',alpha=.2)
fig.savefig(OUT/'paired_effects.pdf',bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(8.8,3.5),layout='constrained')
for variant,label in [('unweighted','Unweighted SVM'),('weighted_1_100','Weighted SVM (1:100)')]:
    d=weights[weights.variant==variant].sort_values('seed')
    ax.plot(d.seed,d.expected_cost,'-o',label=label,markersize=4,linewidth=1.5)
ax.set_xticks([11,22,33,44,55]);ax.set_xlabel('Matched seed')
ax.set_ylabel('Friday expected cost (lower is better)')
ax.set_title('SVM class-weight ablation')
ax.grid(alpha=.22);ax.legend()
fig.savefig(OUT/'svm_weights.pdf',bbox_inches='tight')
plt.close(fig)
for name in ['threshold_ablation.pdf','paired_effects.pdf','svm_weights.pdf']:
    print(f'Regenerated {OUT/name}')
