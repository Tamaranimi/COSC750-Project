from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday"]
DAY_ALIASES = {
    "mon": "monday", "monday": "monday",
    "tue": "tuesday", "tues": "tuesday", "tuesday": "tuesday",
    "wed": "wednesday", "wednesday": "wednesday",
    "thu": "thursday", "thur": "thursday", "thurs": "thursday", "thursday": "thursday",
    "fri": "friday", "friday": "friday",
}


def _normalize_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def _infer_day_from_name(name: str) -> str | None:
    low = name.lower()
    for token, day in DAY_ALIASES.items():
        if re.search(rf"(^|[^a-z]){re.escape(token)}([^a-z]|$)", low):
            return day
    return None


def discover_data_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".csv", ".parquet", ".pq"}])
    if not files:
        raise FileNotFoundError(f"No CSV/Parquet files found recursively under: {root}")
    return files


def _read_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_parquet(path)


def _find_label_col(columns) -> str:
    normalized = {_normalize_col(c): c for c in columns}
    for key in ("label", "labels", "class", "attacklabel", "category"):
        if key in normalized:
            return normalized[key]
    # contains label fallback
    matches = [c for c in columns if "label" in _normalize_col(c)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Could not identify label column. Columns include: {list(columns)[:20]}")


def load_dataset(data_dir: str | Path, attempted_policy: str = "benign") -> tuple[pd.DataFrame, str]:
    """Load all day files, attach source/day metadata, deduplicate, and form binary target.

    attempted_policy: 'benign' or 'exclude'.
    """
    files = discover_data_files(data_dir)
    frames: list[pd.DataFrame] = []
    inferred = []
    for path in files:
        df = _read_one(path)
        day = _infer_day_from_name(path.name)
        inferred.append(day)
        df["__source_file"] = path.name
        df["__day"] = day
        frames.append(df)

    # Fallback for exactly five unlabeled files: sorted file order -> Mon-Fri.
    if len(files) == 5 and all(x is None for x in inferred):
        warnings.warn(
            "Could not infer weekdays from filenames; mapping sorted five files to Monday-Friday. "
            "Verify this mapping before using results in the paper.", RuntimeWarning
        )
        for i, df in enumerate(frames):
            df["__day"] = DAY_ORDER[i]

    full = pd.concat(frames, ignore_index=True, sort=False)
    label_col = _find_label_col(full.columns)

    before = len(full)
    # Deduplicate on original columns, excluding helper metadata so duplicates across files are caught.
    subset_cols = [c for c in full.columns if c not in {"__source_file", "__day"}]
    full = full.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    removed = before - len(full)
    print(f"Loaded {before:,} rows from {len(files)} files; removed {removed:,} exact duplicates.")

    labels = full[label_col].astype(str).str.strip()
    low = labels.str.lower()
    attempted = low.str.contains("attempted", na=False)
    benign = low.str.contains("benign", na=False)
    if attempted_policy == "exclude":
        full = full.loc[~attempted].copy()
        labels = full[label_col].astype(str).str.strip()
        low = labels.str.lower()
        benign = low.str.contains("benign", na=False)
    elif attempted_policy == "benign":
        benign = benign | attempted
    else:
        raise ValueError("attempted_policy must be 'benign' or 'exclude'")

    full["__y"] = (~benign).astype(np.int8)
    return full, label_col


def split_days(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "__day" not in df.columns:
        raise ValueError("Dataset does not contain inferred __day column")
    missing_day = int(df["__day"].isna().sum())
    if missing_day:
        raise ValueError(
            f"{missing_day:,} rows have unknown day. Rename files to include weekday names or edit data.py."
        )
    train = df[df["__day"].isin(["monday", "tuesday", "wednesday"])].copy()
    val = df[df["__day"] == "thursday"].copy()
    test = df[df["__day"] == "friday"].copy()
    if min(len(train), len(val), len(test)) == 0:
        counts = df["__day"].value_counts(dropna=False).to_dict()
        raise ValueError(f"Chronological split produced an empty partition. Day counts: {counts}")
    return train, val, test


def numeric_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    exclude_norm = {
        _normalize_col(label_col), "y", "day", "sourcefile", "timestamp", "flowid",
        "srcip", "sourceip", "dstip", "destinationip", "sourceipv4", "destinationipv4",
    }
    cols = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if _normalize_col(c) in exclude_norm or str(c).startswith("__"):
            continue
        cols.append(c)
    if not cols:
        raise ValueError("No numeric features remain after exclusions")
    return cols


def stratified_sample(df: pd.DataFrame, n_max: int, seed: int) -> pd.DataFrame:
    if len(df) <= n_max:
        return df.copy()
    parts = []
    rng = np.random.default_rng(seed)
    for cls, grp in df.groupby("__y"):
        n = max(1, int(round(n_max * len(grp) / len(df))))
        n = min(n, len(grp))
        parts.append(grp.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n_max:
        out = out.sample(n=n_max, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def sample_at_prevalence(df: pd.DataFrame, prevalence: float, n_max: int, seed: int) -> pd.DataFrame:
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be between 0 and 1")
    pos = df[df["__y"] == 1]
    neg = df[df["__y"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Both benign and attack samples are required")
    feasible_total = min(n_max, int(len(pos) / prevalence), int(len(neg) / (1.0 - prevalence)))
    if feasible_total < 100:
        raise ValueError(
            f"Not enough rows to create a useful sample at prevalence={prevalence}. "
            f"Available attack={len(pos)}, benign={len(neg)}"
        )
    n_pos = max(1, int(round(feasible_total * prevalence)))
    n_neg = max(1, feasible_total - n_pos)
    p = pos.sample(n=min(n_pos, len(pos)), random_state=seed)
    n = neg.sample(n=min(n_neg, len(neg)), random_state=seed + 1)
    return pd.concat([p, n], ignore_index=True).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
