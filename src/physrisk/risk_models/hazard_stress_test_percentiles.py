"""Module containing the HazardPercentilesStressTest class."""


class HazardPercentilesStressTest:
    """Class for storing and managing hazard percentile from the stress test data.

    Attributes
    ----------
        data (dict): Dictionary containing hazard percentile data.

    """

    def __init__(self):
        """Class for storing and managing hazard percentile from the stress test data.

        Attributes
        ----------
            data (dict): Dictionary containing hazard percentile data.

        """
        self.data = {
            "RiverineInundation": {
                "historical_1985": [
                    0,
                    2.0926772356033325,
                    4.517875671386719,
                    8.146644496917732,
                ],
                "rcp4p5_2035": [
                    0,
                    2.123079776763916,
                    4.727473139762878,
                    8.397174167633054,
                ],
                "rcp4p5_2085": [
                    0,
                    2.1849400997161865,
                    4.746373891830444,
                    8.531092262268068,
                ],
                "rcp8p5_2035": [
                    0,
                    2.1317601203918457,
                    4.699397325515747,
                    8.462242317199708,
                ],
                "rcp8p5_2085": [
                    0,
                    2.161303997039795,
                    4.721877813339233,
                    8.560105323791506,
                ],
            },
            "CoastalInundation": {
                "historical_1985": [
                    0,
                    2.3894078731536865,
                    4.403600215911865,
                    5.984654474258423,
                ],
                "rcp45_2050": [
                    0,
                    2.3527116775512695,
                    4.318411707878113,
                    5.894825696945191,
                ],
                "rcp45_2070": [
                    0,
                    2.5329999923706055,
                    4.625603795051575,
                    6.286437940597538,
                ],
                "rcp85_2050": [
                    0,
                    2.318581223487854,
                    4.2181302309036255,
                    5.782120084762571,
                ],
                "rcp85_2070": [
                    0,
                    2.5999999046325684,
                    4.615525245666504,
                    6.324570655822754,
                ],
            },
            "ChronicWind": {
                "historical_1971": [
                    0,
                    2.5999999046325684,
                    4.615525245666504,
                    6.324570655822754,
                ],
                "rcp45_2050": [
                    0,
                    2.5999999046325684,
                    4.615525245666504,
                    6.324570655822754,
                ],
                "rcp45_2100": [
                    0,
                    2.5999999046325684,
                    4.615525245666504,
                    6.324570655822754,
                ],
                "rcp85_2050": [
                    0,
                    2.5999999046325684,
                    4.615525245666504,
                    6.324570655822754,
                ],
                "rcp85_2100": [
                    0,
                    2.5999999046325684,
                    4.615525245666504,
                    6.324570655822754,
                ],
            },
            "Fire": {
                "historical_1971": [
                    0,
                    0.3631645739078522,
                    6.0311747789382935,
                    15.562520790100104,
                ],
                "rcp45_2050": [
                    0,
                    0.5496047139167786,
                    7.531521677970886,
                    17.935127639770517,
                ],
                "rcp45_2100": [
                    0,
                    0.5829123258590698,
                    8.233976364135742,
                    19.678568649292,
                ],
                "rcp85_2050": [
                    0,
                    0.5778544545173645,
                    7.795466423034668,
                    18.597507476806644,
                ],
                "rcp85_2100": [
                    0,
                    1.2540100812911987,
                    10.670519351959229,
                    23.3424015045166,
                ],
            },
            "WaterRisk": {
                "ssp126_2030": [0, 1.0, 5.0, 6.0],
                "ssp126_2050": [0, 1.0, 4.0, 6.0],
                "ssp126_2080": [0, 1.0, 4.0, 6.0],
                "ssp370_2030": [0, 1.0, 5.0, 6.0],
                "ssp370_2050": [0, 1.0, 5.0, 6.0],
                "ssp370_2080": [0, 1.0, 5.0, 6.0],
                "ssp585_2030": [0, 1.0, 5.0, 6.0],
                "ssp585_2050": [0, 1.0, 5.0, 6.0],
                "ssp585_2080": [0, 1.0, 5.0, 6.0],
            },
            "Landslide": {"historical_1980": [0, 2.0, 3.0, 4.0]},
            "Subsidence": {"historical_1980": [0, 2, 3, 4]},
        }

    def get_data(self, hazard_type: str, scenario: str, year: int):
        """Retrieve data for a given hazard type, scenario, and year. This is used in StressTestImpact.

        Args:
        ----
            hazard_type (str): The type of hazard (e.g., 'RiverineInundation').
            scenario (str): The scenario (e.g., 'historical', 'rcp45').
            year (int): The year (e.g., 1971, 2050).

        Returns:
        -------
            list: The data for the given hazard type, scenario, and year.

        """
        key = f"{scenario}_{year}"
        if hazard_type not in self.data:
            return ["Error: Hazard type not found"]

        hazard_data = self.data[hazard_type]

        if key not in hazard_data:
            return ["Error: Data for the given scenario and year not found"]

        return hazard_data[key]


class StressTestImpact:
    """Class for calculating and retrieving stress test impacts for a specific hazard type, scenario, and year."""

    def __init__(
        self,
        hazard_type: type,
        scenario: str,
        year: int,
    ):
        """Initialize a StressTestImpact instance.

        Args:
        ----
            hazard_type (type): The type of hazard associated with the stress test.
            scenario (str): The stress test scenario being considered.
            year (int): The year for which the stress test is performed.

        """
        self.hazard_type = hazard_type
        self.scenario = scenario
        self.year = year
        self.hazard_percentiles = HazardPercentilesStressTest()

    def impact(self):
        """Get the impact data (stress test percentiles) for the given hazard type, scenario, and year.

        Return:
        ------
            Union[ImpactDistrib, EmptyImpactDistrib]: The impact distribution data for the stress test scenario.

        """
        return self.hazard_percentiles.get_data(
            hazard_type=self.hazard_type.__name__,
            scenario=self.scenario,
            year=self.year,
        )
