""" Test asset impact calculations."""
import unittest

import numpy as np

from physrisk import AssetEventDistrib, ExceedanceCurve, RiverineInundation, VulnerabilityDistrib, get_impact_distrib


class TestAssetImpact(unittest.TestCase):
    """Tests asset impact calculations."""

    def test_impact_curve(self):
        """Testing the generation of an asset when only an impact curve (e.g. damage curve is available)"""

        # exceedance curve
        return_periods = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
        exceed_probs = 1.0 / return_periods
        depths = np.array(
            [0.059601218, 0.33267087, 0.50511575, 0.71471703, 0.8641244, 1.0032823, 1.1491022, 1.1634114, 1.1634114]
        )
        curve = ExceedanceCurve(exceed_probs, depths)

        # impact curve
        vul_depths = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1])
        vul_impacts = np.array([0, 1, 2, 7, 14, 30, 60, 180, 365])

        # say we need to add an extra depth point because the damage below that inundation depth is zero
        cutoff_depth = 0.9406518  # 0.75
        curve = curve.add_value_point(cutoff_depth)
        # we could also choose ensure that all impact curve depth points are
        # represented in exceedance curve; we do not here

        depth_bins, probs = curve.get_probability_bins()

        impact_bins = np.interp(depth_bins, vul_depths, vul_impacts)

        include_bin = depth_bins < cutoff_depth
        probs[include_bin[:-1]] = 0  # type: ignore

        mean = np.sum((impact_bins[1:] + impact_bins[:-1]) * probs / 2)  # type: ignore
        self.assertAlmostEqual(mean, 4.8453897)

    def test_protection_level(self):
        return_periods = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
        base_depth = np.array(
            [0.0, 0.22372675, 0.3654859, 0.5393629, 0.6642473, 0.78564394, 0.9406518, 1.0539534, 1.1634114]
        )
        # future_depth = np.array(
        #     [0.059601218, 0.33267087, 0.50511575, 0.71471703, 0.8641244, 1.0032823, 1.1491022, 1.1634114, 1.1634114]
        # )

        exceed_probs = 1.0 / return_periods

        protection_return_period = 250.0  # protection level of 250 years
        protection_depth = np.interp(1.0 / protection_return_period, exceed_probs[::-1], base_depth[::-1])

        self.assertAlmostEqual(protection_depth, 0.9406518)  # type: ignore

    def test_single_asset_impact(self):
        # exceedance curve
        return_periods = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
        exceed_probs = 1.0 / return_periods
        depths = np.array(
            [0.059601218, 0.33267087, 0.50511575, 0.71471703, 0.8641244, 1.0032823, 1.1491022, 1.1634114, 1.1634114]
        )
        curve = ExceedanceCurve(exceed_probs, depths)

        cutoff_depth = 0.9406518
        curve = curve.add_value_point(cutoff_depth)

        # impact curve
        vul_depths = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1])
        vul_impacts = np.array([0, 1, 2, 7, 14, 30, 60, 180, 365])

        depth_bins, probs = curve.get_probability_bins()

        impact_bins = np.interp(depth_bins, vul_depths, vul_impacts)

        # if upper end of bin less then cutoff then exclude
        probs_w_cutoff = np.where(depth_bins[1:] <= cutoff_depth, 0.0, 1.0)
        # n_bins = len(probs)  # type: ignore
        vul = VulnerabilityDistrib(
            type(RiverineInundation), depth_bins, impact_bins, np.diag(probs_w_cutoff)
        )  # np.eye(n_bins, n_bins))
        event = AssetEventDistrib(type(RiverineInundation), depth_bins, probs)  # type: ignore

        impact = get_impact_distrib(event, vul)
        mean = impact.mean_impact()

        self.assertAlmostEqual(mean, 4.8453897)
