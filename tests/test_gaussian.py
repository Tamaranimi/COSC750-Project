import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids_stage3.models import DiagonalGaussianBayes


def test_gaussian_bayes_orders_simple_clusters():
    X = np.array([[0., 0.], [0.1, -0.1], [3., 3.], [3.2, 2.9]])
    y = np.array([0, 0, 1, 1])
    model = DiagonalGaussianBayes(var_floor=1e-4).fit(X, y)
    p = model.predict_proba(np.array([[0., 0.], [3., 3.]]))[:, 1]
    assert p[0] < 0.5
    assert p[1] > 0.5
