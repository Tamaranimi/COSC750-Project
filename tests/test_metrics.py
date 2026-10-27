import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids_stage3.metrics import expected_cost, best_threshold_by_cost


def test_expected_cost():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 0, 1])
    # one FP and one FN => (1 + 100) / 4
    assert abs(expected_cost(y, pred, 1, 100) - 25.25) < 1e-12


def test_threshold_returns_valid_value():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.01, 0.1, 0.8, 0.9])
    t, cost = best_threshold_by_cost(y, s, 1, 100)
    assert 0 <= t <= 1
    assert cost >= 0
