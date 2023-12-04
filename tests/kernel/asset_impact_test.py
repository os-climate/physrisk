""" Test asset impact calculations."""
import unittest

import numpy as np

from physrisk.kernel.assets import RealEstateAsset
from physrisk.kernel.curve import ExceedanceCurve
from physrisk.kernel.hazard_event_distrib import HazardEventDistrib
from physrisk.kernel.hazard_model import HazardDataRequest
from physrisk.kernel.hazards import RiverineInundation
from physrisk.kernel.impact import ImpactDistrib
from physrisk.kernel.vulnerability_distrib import VulnerabilityDistrib
from physrisk.vulnerability_models.real_estate_models import (
    RealEstateCoastalInundationModel,
    RealEstateRiverineInundationModel,
)


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
        event = HazardEventDistrib(type(RiverineInundation), depth_bins, probs)  # type: ignore

        impact_prob = vul.prob_matrix.T @ event.prob
        impact = ImpactDistrib(vul.event_type, vul.impact_bins, impact_prob)

        mean = impact.mean_impact()

        self.assertAlmostEqual(mean, 4.8453897)

    def test_performance_hazardlookup(self):
        """Just for reference: not true test"""
        asset_requests = {}
        import time

        start = time.time()

        assets = [RealEstateAsset(latitude=0, longitude=0, location="", type="") for _ in range(10000)]

        vulnerability_models = [RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()]

        time_assets = time.time() - start
        print(f"Time for asset generation {time_assets}s ")
        start = time.time()
        # we key requests via model and assets; let's check dictionary look-up is fast enough
        # (there are less simple alternatives)

        # create requests:
        for v in vulnerability_models:
            for a in assets:
                asset_requests[(v, a)] = [
                    HazardDataRequest(RiverineInundation, 0, 0, indicator_id="", scenario="", year=2030)
                ]

        time_requests = time.time() - start
        print(f"Time for requests dictionary creation {time_requests}s ")
        start = time.time()
        # read requests:
        for key in asset_requests:
            if asset_requests[key][0].longitude != 0:
                raise Exception()

        time_responses = time.time() - start
        print(f"Time for response dictionary creation {time_responses}s ")
