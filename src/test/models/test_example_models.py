import unittest

import numpy as np

from physrisk.data.data_requests import EventDataResponse
from physrisk.kernel.assets import Asset
from physrisk.kernel.events import Inundation
from physrisk.models.example_models import ExampleCdfBasedVulnerabilityModel


class TestExampleModels(unittest.TestCase):
    def test_pdf_based_vulnerability_model(self):
        model = ExampleCdfBasedVulnerabilityModel(model="", event_type=Inundation)

        latitude, longitude = 45.268405, 19.885738
        asset = Asset(latitude, longitude)

        return_periods = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
        intensities = np.array(
            [0.059601218, 0.33267087, 0.50511575, 0.71471703, 0.8641244, 1.0032823, 1.1491022, 1.1634114, 1.1634114]
        )

        mock_response = EventDataResponse(return_periods, intensities)

        vul, event = model.get_distributions(asset, [mock_response])

        print(vul)
