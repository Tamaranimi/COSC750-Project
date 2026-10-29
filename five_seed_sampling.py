import numpy as np
from temporal_blocks import blocks
from sampling_config import (
    MAX_TOTAL_RECORDS,
    MIN_FIT_ATTACKS,
    MIN_CAL_ATTACKS,
    MIN_FIT_BLOCKS,
    MIN_FIT_FAMILIES,
)

def sample_fit_calibration(df, seed, max_attempts=2000):
    """Find reproducible, disjoint fitting/calibration temporal blocks."""
    ids = blocks(df)
    label_col = "Label"

    if label_col not in df.columns:
        raise ValueError("Original attack-family Label column required")

    temp = df[[label_col, "__y"]].copy()
    temp["__block"] = ids

    summary = temp.groupby("__block")["__y"].agg(
        total="size", attacks="sum"
    )

    eligible = summary[
        (summary["attacks"] >= 100) &
        (summary["total"] <= MAX_TOTAL_RECORDS // 2)
    ]

    if len(eligible) < 4:
        raise ValueError("Insufficient eligible temporal blocks")

    rng = np.random.default_rng(seed)
    candidates = eligible.index.to_numpy()

    for _ in range(max_attempts):
        order = rng.permutation(candidates)
        chosen = []
        total = 0

        for block in order:
            size = int(summary.loc[block, "total"])
            if total + size <= MAX_TOTAL_RECORDS:
                chosen.append(block)
                total += size

        if len(chosen) < MIN_FIT_BLOCKS + 1:
            continue

        cal_block = chosen[-1]
        fit_blocks = chosen[:-1]

        fit = df.loc[np.isin(ids, fit_blocks)].copy()
        cal = df.loc[ids == cal_block].copy()

        if int(fit["__y"].sum()) < MIN_FIT_ATTACKS:
            continue

        if int(cal["__y"].sum()) < MIN_CAL_ATTACKS:
            continue

        if len(fit_blocks) < MIN_FIT_BLOCKS:
            continue

        families = set(
            fit.loc[fit["__y"] == 1, label_col].astype(str).str.strip()
        )

        if len(families) < MIN_FIT_FAMILIES:
            continue

        if fit["__y"].nunique() != 2 or cal["__y"].nunique() != 2:
            continue

        if not set(blocks(fit)).isdisjoint(set(blocks(cal))):
            continue

        return fit, cal

    raise ValueError(
        f"Seed {seed}: no suitable allocation after {max_attempts} attempts"
    )
