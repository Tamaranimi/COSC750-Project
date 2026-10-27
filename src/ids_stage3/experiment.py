from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import psutil
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import (
    load_dataset,
    numeric_feature_columns,
    sample_at_prevalence,
    split_days,
    stratified_sample,
)
from .metrics import best_threshold_by_cost, binary_metrics
from .models import DiagonalGaussianBayes


def _prepare_arrays(train_df, val_df, test_df, feature_cols):
    Xtr = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    Xv = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    Xte = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    ytr = train_df["__y"].to_numpy(dtype=int)
    yv = val_df["__y"].to_numpy(dtype=int)
    yte = test_df["__y"].to_numpy(dtype=int)

    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    Xtr = prep.fit_transform(Xtr)
    Xv = prep.transform(Xv)
    Xte = prep.transform(Xte)
    return Xtr, ytr, Xv, yv, Xte, yte


def _row(method, seed, metrics, train_seconds, threshold=np.nan, note="", params=None):
    return {
        "method": method,
        "seed": seed,
        **metrics,
        "train_seconds": float(train_seconds),
        "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
        "params": json.dumps(params or {}, sort_keys=True),
        "note": note,
    }


def run_experiment(
    data_dir: str,
    output_dir: str,
    seeds=(11, 22, 33, 44, 55),
    target_prevalence: float = 0.01,
    c_fp: float = 1.0,
    c_fn: float = 100.0,
    max_train: int = 12000,
    max_val: int = 10000,
    max_test: int = 25000,
    attempted_policy: str = "benign",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full, label_col = load_dataset(data_dir, attempted_policy=attempted_policy)
    train_full, val_full, test_full = split_days(full)
    feature_cols = numeric_feature_columns(full, label_col)

    split_summary = {
        "rows_after_dedup": int(len(full)),
        "train_mon_wed": int(len(train_full)),
        "validation_thu": int(len(val_full)),
        "test_fri": int(len(test_full)),
        "numeric_features_used": int(len(feature_cols)),
        "target_prevalence": target_prevalence,
        "c_fp": c_fp,
        "c_fn": c_fn,
        "seeds": list(seeds),
        "attempted_policy": attempted_policy,
    }

    rows = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        train_df = stratified_sample(train_full, max_train, seed)
        val_df = sample_at_prevalence(val_full, target_prevalence, max_val, seed + 1000)
        test_df = sample_at_prevalence(test_full, target_prevalence, max_test, seed + 2000)
        Xtr, ytr, Xv, yv, Xte, yte = _prepare_arrays(train_df, val_df, test_df, feature_cols)

        # B3: trivial always-pass floor.
        pred = np.zeros_like(yte)
        rows.append(_row("B3 Always-pass", seed, binary_metrics(yte, pred, c_fp=c_fp, c_fn=c_fn), 0.0,
                         note="Trivial floor; predicts benign for every flow"))

        # B1: classical Gaussian Bayes with MAP threshold 0.5.
        t0 = time.perf_counter()
        gb = DiagonalGaussianBayes(var_floor=1e-6).fit(Xtr, ytr)
        gb_time = time.perf_counter() - t0
        gb_score = gb.predict_proba(Xte, attack_prior=target_prevalence)[:, 1]
        gb_pred = (gb_score >= 0.5).astype(int)
        rows.append(_row("B1 Gaussian Bayes (MAP)", seed,
                         binary_metrics(yte, gb_pred, score=gb_score, c_fp=c_fp, c_fn=c_fn), gb_time,
                         threshold=0.5, note="Own NumPy diagonal-Gaussian implementation",
                         params={"var_floor": 1e-6, "attack_prior": target_prevalence}))

        # Proposed preliminary method: same density estimator + Bayes minimum-risk threshold.
        risk_threshold = c_fp / (c_fp + c_fn)
        prop_pred = (gb_score >= risk_threshold).astype(int)
        rows.append(_row("Proposed Gaussian Bayes + min-risk", seed,
                         binary_metrics(yte, prop_pred, score=gb_score, c_fp=c_fp, c_fn=c_fn), gb_time,
                         threshold=risk_threshold, note="Preliminary proposed-method result",
                         params={"var_floor": 1e-6, "attack_prior": target_prevalence,
                                 "c_fp": c_fp, "c_fn": c_fn}))

        # B2: cost-weighted RBF SVM. Hyperparameters chosen by CV on training only.
        base = SVC(kernel="rbf", class_weight={0: 1.0, 1: c_fn}, probability=False, cache_size=2048)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        grid = GridSearchCV(
            base,
            param_grid={"C": [1.0, 10.0], "gamma": ["scale"]},
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        t0 = time.perf_counter()
        grid.fit(Xtr, ytr)
        best_svc = SVC(
            kernel="rbf",
            class_weight={0: 1.0, 1: c_fn},
            probability=False,
            cache_size=2048,
            C=grid.best_params_["C"],
            gamma=grid.best_params_["gamma"],
        )
        calibrated = CalibratedClassifierCV(estimator=best_svc, method="sigmoid", cv=3, n_jobs=-1)
        calibrated.fit(Xtr, ytr)
        svm_time = time.perf_counter() - t0
        val_score = calibrated.predict_proba(Xv)[:, 1]
        threshold, val_cost = best_threshold_by_cost(yv, val_score, c_fp=c_fp, c_fn=c_fn)
        test_score = calibrated.predict_proba(Xte)[:, 1]
        test_pred = (test_score >= threshold).astype(int)
        rows.append(_row("B2 Cost-weighted RBF SVM", seed,
                         binary_metrics(yte, test_pred, score=test_score, c_fp=c_fp, c_fn=c_fn), svm_time,
                         threshold=threshold, note=f"Threshold chosen on Thursday validation only; val_cost={val_cost:.6g}",
                         params={**grid.best_params_, "class_weight_attack": c_fn, "calibration": "sigmoid"}))

    runs = pd.DataFrame(rows)
    metric_cols = ["expected_cost", "attack_recall", "fpr", "macro_f1", "pr_auc", "train_seconds"]
    summary_rows = []
    for method, grp in runs.groupby("method", sort=False):
        item = {"method": method, "n_seeds": int(grp["seed"].nunique())}
        for col in metric_cols:
            item[f"{col}_mean"] = float(grp[col].mean())
            item[f"{col}_std"] = float(grp[col].std(ddof=1)) if len(grp) > 1 else 0.0
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows)

    runs.to_csv(out_dir / "baseline_runs.csv", index=False)
    summary.to_csv(out_dir / "baseline_summary.csv", index=False)
    _write_tex_tables(summary, out_dir)

    metadata = {
        **split_summary,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "feature_columns": feature_cols,
        "IMPORTANT": "Test set is Friday only. Do not tune using baseline_runs.csv test metrics.",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nSummary:\n", summary.to_string(index=False))
    print(f"\nWrote results to: {out_dir.resolve()}")
    return runs, summary


def _fmt(mean, std, digits=3):
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _write_tex_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    lookup = {r["method"]: r for _, r in summary.iterrows()}
    order = ["B3 Always-pass", "B1 Gaussian Bayes (MAP)", "B2 Cost-weighted RBF SVM", "Proposed Gaussian Bayes + min-risk"]
    label = {
        "B3 Always-pass": "B3 always-pass",
        "B1 Gaussian Bayes (MAP)": "B1 Gaussian Bayes (MAP)",
        "B2 Cost-weighted RBF SVM": "B2 cost-weighted RBF SVM",
        "Proposed Gaussian Bayes + min-risk": "Proposed Gaussian min-risk",
    }
    lines = []
    for method in order:
        r = lookup.get(method)
        if r is None:
            continue
        lines.append(
            f"{label[method]} & {_fmt(r['expected_cost_mean'], r['expected_cost_std'], 4)} & "
            f"{_fmt(r['attack_recall_mean'], r['attack_recall_std'], 3)} & "
            f"{_fmt(r['macro_f1_mean'], r['macro_f1_std'], 3)} & "
            f"{_fmt(r['train_seconds_mean'], r['train_seconds_std'], 1)} \\\\" 
        )
    (out_dir / "baseline_table_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "baseline_table_beamer_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
