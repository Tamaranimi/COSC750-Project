import numpy as np
from run_stage4_ablation import threshold_variants


def test_threshold_variants_have_expected_fixed_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.001, 0.01, 0.1, 0.9])
    thresholds = threshold_variants(y, p, 0.01, 1, 100)
    assert thresholds['fixed_0_5'] == 0.5
    assert np.isclose(thresholds['bayes_cost_threshold'], 1 / 101)
    assert 0 <= thresholds['thursday_optimized'] <= 1
