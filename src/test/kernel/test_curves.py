""" Test asset impact calculations."""
import unittest
import numpy as np
from physrisk import AssetEventDistribution, ExceedanceCurve, VulnerabilityDistrib
from physrisk import Drought, Inundation
from physrisk import get_impact_distribution

class TestAssetImpact(unittest.TestCase):
    """Tests asset impact calculations."""

    def test_return_period_data(self):
        return_periods = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
        depths = np.array([0.059, 0.33, 0.51, 0.71, 0.86, 1.00, 1.15, 1.16, 1.16])
        
        # say we need to add an extra depth point because the damage below that point is zero
        extra_depth = 0.75

        exceed_probs = 1.0 / return_periods
        curve = ExceedanceCurve(exceed_probs, depths)
        curve = curve.add_value_point(0.75)

        self.assertAlmostEqual(curve.probs[4], 0.03466667)

        


        