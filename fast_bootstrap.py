import numpy as np

def bootstrap_blocks_fast(y, score, threshold, prior, c_fp, c_fn,
                          group_ids, seed=123, n_boot=200):
    """Bootstrap whole temporal blocks using precomputed confusion counts."""
    y = np.asarray(y)
    score = np.asarray(score)
    group_ids = np.asarray(group_ids)

    if not (len(y) == len(score) == len(group_ids)):
        raise ValueError("Input lengths must match")

    unique, inverse = np.unique(group_ids, return_inverse=True)
    n = len(unique)

    predicted = score >= threshold
    benign = y == 0
    attacks = y == 1

    fp = np.bincount(inverse, weights=(benign & predicted), minlength=n)
    fn = np.bincount(inverse, weights=(attacks & ~predicted), minlength=n)
    n0 = np.bincount(inverse, weights=benign, minlength=n)
    n1 = np.bincount(inverse, weights=attacks, minlength=n)

    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(n_boot):
        draw = rng.integers(0, n, size=n)

        total0 = n0[draw].sum()
        total1 = n1[draw].sum()

        if total0 == 0 or total1 == 0:
            continue

        fpr = fp[draw].sum() / total0
        fnr = fn[draw].sum() / total1

        vals.append((1-prior)*c_fp*fpr + prior*c_fn*fnr)

    if len(vals) < 20:
        return (np.nan, np.nan, len(vals))

    low, high = np.quantile(vals, [0.025, 0.975])
    return float(low), float(high), len(vals)
