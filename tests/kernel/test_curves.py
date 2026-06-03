"""Test asset impact calculations."""

import numpy as np
import pytest

from physrisk.kernel.curve import ExceedanceCurve


def test_return_period_data():
    return_periods = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
    depths = np.array([0.059, 0.33, 0.51, 0.71, 0.86, 1.00, 1.15, 1.16, 1.16])

    exceedance_probs = 1.0 / return_periods
    curve = ExceedanceCurve(exceedance_probs, depths).add_value_point(0.75)

    assert curve.probs[4] == pytest.approx(0.03466667, rel=1e-6)
