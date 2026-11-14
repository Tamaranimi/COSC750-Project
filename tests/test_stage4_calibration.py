import numpy as np
from scipy.special import expit
from run_stage4_calibration import score_to_uncalibrated_probability, model_variants


def test_uncalibrated_svm_heuristic():
    scores = np.array([-2., 0., 2.])
    assert np.allclose(score_to_uncalibrated_probability(scores, 'SVM'), expit(scores))


def test_platt_and_uncalibrated_variants_exist():
    cal = np.array([-3., -2., -1., 1., 2., 3.])
    labels = np.array([0, 0, 0, 1, 1, 1])
    variants = model_variants(cal, np.array([-1., 1.]), np.array([-2., 2.]), labels, 'SVM')
    assert set(variants) == {'uncalibrated', 'platt_sigmoid'}
    for pv, pt in variants.values():
        assert pv.shape == (2,) and pt.shape == (2,)
        assert np.all((pv >= 0) & (pv <= 1))
