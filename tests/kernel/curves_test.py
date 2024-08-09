"""Test asset impact calculations."""

import numpy as np
import pytest

from physrisk.kernel.curve import ExceedanceCurve


@pytest.fixture
def curve_data():
    return_periods = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
    depths = np.array([0.059, 0.33, 0.51, 0.71, 0.86, 1.00, 1.15, 1.16, 1.16])
    exceed_probs = 1.0 / return_periods
    return ExceedanceCurve(exceed_probs, depths)


def test_return_period_data(curve_data):
    curve = curve_data.add_value_point(0.75)
    assert pytest.approx(curve.probs[4], rel=1e-6) == 0.03466667
