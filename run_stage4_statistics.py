"""Stage 4 paired ablation statistics and figures; no model retraining.

Run from the COSC750-Project repository root.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMPARISONS = [
    ('Gaussian', 'fixed_0_5', 'thursday_optimized', 'Threshold: Gaussian fixed vs Thursday'),
    ('Gaussian', 'bayes_cost_threshold', 'thursday_optimized', 'Threshold: Gaussian Bayes vs Thursday'),
    ('SVM', 'fixed_0_5', 'thursday_optimized', 'Threshold: SVM fixed vs Thursday'),
    ('SVM', 'bayes_cost_threshold', 'thursday_optimized', 'Threshold: SVM Bayes vs Thursday'),
    ('SVM', 'weighted_1_100', 'unweighted', 'SVM class weight: weighted vs unweighted'),
]

def paired_test(a, b):
    """Positive difference means the second (b) configuration costs less."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    if n < 3:
        raise ValueError('At least three matched seeds required')
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if np.isclose(sd, 0, rtol=0, atol=1e-13):
        if abs(mean) < 1e-13:
            return dict(n=n, mean_difference=0.0, sd_difference=0.0, ci_low=0.0, ci_high=0.0,
                        t_statistic=0.0, p_value=1.0, note='All paired differences zero')
        return dict(n=n, mean_difference=mean, sd_difference=0.0, ci_low=mean, ci_high=mean,
                    t_statistic=np.nan, p_value=np.nan, note='Constant nonzero differences; t-test undefined')
    se = sd / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
    t, p = stats.ttest_1samp(d, 0)
    return dict(n=n, mean_difference=mean, sd_difference=sd, ci_low=float(lo), ci_high=float(hi),
                t_statistic=float(t), p_value=float(p), note='Exploratory paired t-test')

def holm_adjust(pvalues):
    """Holm step-down adjustment; NaN values remain NaN."""
    arr = np.asarray(pvalues, dtype=float)
    out = np.full(len(arr), np.nan)
    idx = np.where(np.isfinite(arr))[0]
    sorted_idx = idx[np.argsort(arr[idx])]
    prev = 0.0
    m = len(sorted_idx)
    for rank, i in enumerate(sorted_idx):
        prev = max(prev, min(1.0, float(arr[i]) * (m - rank)))
        out[i] = prev
    return out

def load_pairs(df, group_col, group_value, first, second):
    part = df.loc[df[group_col] == group_value] if group_col else df
    part = part.loc[part['variant'].isin([first, second])]
    if part.duplicated(['seed', 'variant']).any():
        raise ValueError(f'Duplicate seed/variant rows for {group_value}')
    wide = part.pivot(index='seed', columns='variant', values='expected_cost')
    if first not in wide or second not in wide:
        raise ValueError(f'Missing variants {first}, {second} for {group_value}')
    if wide[[first, second]].isna().any().any():
        raise ValueError(f'Unmatched seed values for {group_value}: {wide}')
    if len(wide) != 5:
        raise ValueError(f'Expected 5 matched seeds; got {len(wide)} for {group_value}')
    return wide.sort_index()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold-csv', default='stage4_ablation_results/ablation_runs.csv')
    ap.add_argument('--weight-csv', default='stage4_svm_weight_results/svm_weight_runs.csv')
    ap.add_argument('--output-dir', default='stage4_statistical_results')
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = pd.read_csv(args.threshold_csv)
    weights = pd.read_csv(args.weight_csv)
    results = []
    paired_rows = []
    for model, first, second, label in COMPARISONS:
        source = weights if label.startswith('SVM class weight:') else thresholds
        wide = load_pairs(source, None if source is weights else 'model', model, first, second)
        result = paired_test(wide[first], wide[second])
        results.append(dict(comparison=label, model=model, first=first, second=second,
                            mean_cost_first=float(wide[first].mean()),
                            mean_cost_second=float(wide[second].mean()), **result))
        for seed, row in wide.iterrows():
            paired_rows.append(dict(comparison=label, seed=int(seed), first_cost=float(row[first]),
                                    second_cost=float(row[second]), difference=float(row[first] - row[second])))
    summary = pd.DataFrame(results)
    summary['p_holm_5'] = holm_adjust(summary['p_value'])
    summary['significant_holm_0_05'] = summary['p_holm_5'] < 0.05
    summary.to_csv(out / 'ablation_paired_statistics.csv', index=False)
    pd.DataFrame(paired_rows).to_csv(out / 'ablation_paired_seed_differences.csv', index=False)

    # Figure 1: paired cost comparison, seed by seed (no uncertainty concealed by bars).
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, model in zip(axes, ['Gaussian', 'SVM']):
        subset = thresholds.loc[thresholds.model == model]
        for variant, data in subset.groupby('variant'):
            data = data.sort_values('seed')
            ax.plot(data.seed, data.expected_cost, marker='o', label={'bayes_cost_threshold': 'Bayesian cost threshold', 'fixed_0_5': 'Fixed threshold (0.5)', 'thursday_optimized': 'Thursday-optimized threshold'}.get(variant, variant))
        ax.set(title=model, xlabel='Random seed', ylabel='Friday expected cost')
        ax.set_xticks(sorted(subset.seed.unique()))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle('Threshold ablation: five matched seeds (1% attack prevalence)')
    fig.savefig(out / 'figure_threshold_ablation.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    for variant, data in weights.groupby('variant'):
        data = data.sort_values('seed')
        ax.plot(data.seed, data.expected_cost, marker='o', label={'unweighted': 'Unweighted SVM', 'weighted_1_100': 'Weighted SVM (1:100)'}.get(variant, variant))
    ax.set(title='SVM class-weight ablation: five matched seeds', xlabel='Random seed',
           ylabel='Friday expected cost')
    ax.set_xticks(sorted(weights.seed.unique()))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out / 'figure_svm_class_weights.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    positions = np.arange(len(summary))
    ax.errorbar(summary.mean_difference, positions,
                xerr=[summary.mean_difference - summary.ci_low,
                      summary.ci_high - summary.mean_difference], fmt='o', capsize=4)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(positions, summary.comparison)
    ax.invert_yaxis()
    ax.set(xlabel='Mean paired cost difference (first minus second), 95% t CI',
           title='Ablation comparisons (five paired seeds)')
    ax.grid(axis='x', alpha=0.25)
    fig.savefig(out / 'figure_ablation_paired_effects.png', dpi=220)
    plt.close(fig)

    (out / 'analysis_protocol.json').write_text(json.dumps({
        'test': 'Two-sided one-sample t-test of paired seed differences',
        'confidence_interval': '95% Student t interval, df=4',
        'multiplicity': 'Holm adjustment across five exploratory ablation comparisons',
        'difference': 'cost(first) - cost(second); positive favors second',
        'limitation': 'Five random seeds share one temporal dataset; independence and normality are not established',
        'zero_difference_policy': 'Exactly zero paired differences: p=1; constant nonzero differences: t-test undefined',
        'model_training': 'None; reads existing CSVs only',
        'inputs': [args.threshold_csv, args.weight_csv],
    }, indent=2), encoding='utf-8')
    print(summary[['comparison', 'mean_difference', 'ci_low', 'ci_high', 'p_value', 'p_holm_5']].to_string(index=False))
    print(f'\nSaved statistical CSVs, protocol, and three PNG figures in {out.resolve()}')

if __name__ == '__main__':
    main()
