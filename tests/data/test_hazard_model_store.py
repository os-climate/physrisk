import os
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import zarr
import zarr.storage
from affine import Affine
from pyproj import Transformer

from physrisk.hazard_models.core_hazards import cmip6_scenario_to_rcp


class TestData:
    # fmt: off
    longitudes = [69.4787, 68.71, 20.1047, 19.8936, 19.6359, 0.5407, 6.9366, 6.935, 13.7319, 13.7319, 14.4809, -68.3556, -68.3556, -68.9892, -70.9157] # noqa
    latitudes = [34.556, 35.9416, 39.9116, 41.6796, 42.0137, 35.7835, 36.8789, 36.88, -12.4706, -12.4706, -9.7523, -38.9368, -38.9368, -34.5792, -39.2145] # noqa

    coastal_longitudes = [12.2, 50.5919, 90.3473, 90.4295, 90.4804, 90.3429, 90.5153, 90.6007]
    coastal_latitudes = [-5.55, 26.1981, 23.6473, 23.6783, 23.5699, 23.9904, 23.59, 23.6112]
    # fmt: on

    # fmt: off
    wind_return_periods = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] # noqa
    wind_intensities_1 = [34.314999, 40.843750, 44.605000, 46.973751, 48.548752, 49.803749, 51.188751, 52.213749, 52.902500, 53.576248, 57.552502, 59.863750, 60.916248, 61.801250, 62.508751, 63.082500, 63.251251, 63.884998, 64.577499] # noqa
    wind_intensities_2 = [37.472500, 44.993752, 49.049999, 51.957500, 53.796249, 55.478748, 56.567501, 57.572498, 58.661251, 59.448750, 63.724998, 65.940002, 66.842499, 67.614998, 68.110001, 68.547501, 68.807503, 69.529999, 70.932503] # noqa
    # fmt: on

    # fmt: off
    temperature_thresholds = [10, 20, 30, 40, 50] # noqa
    degree_days_above_index_1 = [6000, 3000, 100, 20, 10] # noqa
    degree_days_above_index_2 = [7000, 4000, 150, 30, 12] # noqa
    # fmt: on


class ZarrStoreMocker:
    def __init__(self):
        self.store, self._root = zarr_memory_store()

    def add_curves_global(
        self,
        array_path: str,
        longitudes: Sequence[float],
        latitudes: Sequence[float],
        index_values: Union[List[float], npt.NDArray, List[str]],
        intensities: Union[List[float], npt.NDArray],
        width: int = 43200,
        height: int = 21600,
        units: str = "default",
    ):
        crs = "epsg:4326"
        crs, shape, trans = self._crs_shape_transform_global(
            index_values=index_values, width=width, height=height
        )
        self._add_curves(
            array_path,
            longitudes,
            latitudes,
            crs,
            shape,
            trans,
            index_values,
            intensities,
            units=units,
        )

    def _crs_shape_transform_global(
        self,
        width: int = 43200,
        height: int = 21600,
        index_values: Union[List[float], npt.NDArray, List[str]] = [0.0],
    ):
        return self._crs_shape_transform(width, height, index_values)

    def _add_curves(
        self,
        array_path: str,
        longitudes: Sequence[float],
        latitudes: Sequence[float],
        crs: str,
        shape: Tuple[int, int, int],
        trans: List[float],
        index_values: Union[Sequence[float], npt.NDArray, List[str]],
        intensities: Union[Sequence[float], npt.NDArray],
        units: str = "default",
    ):
        z = self._root.create_dataset(  # type: ignore
            array_path,
            shape=(shape[0], shape[1], shape[2]),
            chunks=(shape[0], 1000, 1000),
            dtype="f4",
        )
        z.attrs["transform_mat3x3"] = trans
        z.attrs["index_values"] = index_values
        z.attrs["crs"] = crs
        z.attrs["units"] = units

        if crs.lower() != "epsg:4326":
            transproj = Transformer.from_crs(
                "epsg:4326",
                crs,
                always_xy=True,
            )
            x, y = transproj.transform(longitudes, latitudes)
        else:
            x, y = longitudes, latitudes

        transform = Affine(trans[0], trans[1], trans[2], trans[3], trans[4], trans[5])
        coords = np.vstack((x, y, np.ones(len(x))))
        inv_trans = ~transform
        mat = np.array(inv_trans).reshape(3, 3)
        frac_image_coords = mat @ coords
        image_coords = np.floor(frac_image_coords).astype(int)
        if isinstance(intensities, np.ndarray) and len(np.array(intensities).shape) > 1:
            for j in range(len(x)):
                z[:, image_coords[1, j], image_coords[0, j]] = intensities[j, :]
        else:
            for j in range(len(x)):
                z[:, image_coords[1, j], image_coords[0, j]] = intensities

    def _crs_shape_transform(
        self,
        width: int,
        height: int,
        index_values: Union[List[float], npt.NDArray, List[str]] = [0.0],
    ):
        t = [360.0 / width, 0.0, -180.0, 0.0, -180.0 / height, 90.0, 0.0, 0.0, 1.0]
        return "epsg:4326", (len(index_values), height, width), t


def shape_transform_21600_43200(
    width: int = 43200,
    height: int = 21600,
    return_periods: Union[List[float], npt.NDArray] = [0.0],
):
    t = [360.0 / width, 0.0, -180.0, 0.0, -180.0 / height, 90.0, 0.0, 0.0, 1.0]
    return (len(return_periods), height, width), t


def zarr_memory_store(path="hazard.zarr"):
    store = zarr.storage.MemoryStore(root=path)
    return store, zarr.open(store=store, mode="w")


def add_curves(
    root: zarr.Group,
    longitudes,
    latitudes,
    array_path: str,
    shape: Tuple[int, int, int],
    curve: np.ndarray,
    return_periods: List[float],
    trans: List[float],
):
    z = root.create_dataset(  # type: ignore
        array_path,
        shape=(shape[0], shape[1], shape[2]),
        chunks=(shape[0], 1000, 1000),
        dtype="f4",
    )
    z.attrs["transform_mat3x3"] = trans
    z.attrs["index_values"] = return_periods

    trans = z.attrs["transform_mat3x3"]
    transform = Affine(trans[0], trans[1], trans[2], trans[3], trans[4], trans[5])

    coords = np.vstack((longitudes, latitudes, np.ones(len(longitudes))))
    inv_trans = ~transform
    mat = np.array(inv_trans).reshape(3, 3)
    frac_image_coords = mat @ coords
    image_coords = np.floor(frac_image_coords).astype(int)
    for j in range(len(longitudes)):
        z[:, image_coords[1, j], image_coords[0, j]] = curve


def get_mock_hazard_model_store_single_curve():
    """Create a test MemoryStore for creation of Zarr hazard model for unit testing. A single curve
    is applied at all locations."""

    return_periods = inundation_return_periods()
    t = [
        0.008333333333333333,
        0.0,
        -180.0,
        0.0,
        -0.008333333333333333,
        90.0,
        0.0,
        0.0,
        1.0,
    ]
    shape = (len(return_periods), 21600, 43200)
    store = zarr.storage.MemoryStore(root="hazard.zarr")
    root = zarr.open(store=store, mode="w")
    array_path = get_source_path_wri_riverine_inundation(
        model="MIROC-ESM-CHEM", scenario="rcp8p5", year=2080
    )
    z = root.create_dataset(  # type: ignore
        array_path,
        shape=(shape[0], shape[1], shape[2]),
        chunks=(shape[0], 1000, 1000),
        dtype="f4",
    )
    z.attrs["transform_mat3x3"] = t
    z.attrs["index_values"] = return_periods

    longitudes = TestData.longitudes
    latitudes = TestData.latitudes
    t = z.attrs["transform_mat3x3"]
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

    coords = np.vstack((longitudes, latitudes, np.ones(len(longitudes))))
    inv_trans = ~transform
    mat = np.array(inv_trans).reshape(3, 3)
    frac_image_coords = mat @ coords
    image_coords = np.floor(frac_image_coords).astype(int)
    z[:, image_coords[1, 1], image_coords[0, 1]] = np.linspace(0.1, 1.0, z.shape[0])

    return store


def mock_hazard_model_store_heat(longitudes, latitudes):
    return mock_hazard_model_store_for_parameter_sets(
        longitudes, latitudes, degree_day_heat_parameter_set()
    )


def mock_hazard_model_store_heat_wbgt(longitudes, latitudes):
    return mock_hazard_model_store_for_parameter_sets(
        longitudes, latitudes, wbgt_gzn_joint_parameter_set()
    )


def mock_hazard_model_store_inundation(longitudes, latitudes, curve):
    return mock_hazard_model_store_single_curve_for_paths(
        longitudes, latitudes, curve, inundation_paths
    )


def mock_hazard_model_store_for_parameter_sets(longitudes, latitudes, path_parameters):
    """Create a MemoryStore for creation of Zarr hazard model to be used with unit tests,
    with the specified longitudes and latitudes set to the curve supplied."""

    return_periods = None
    shape = (1, 21600, 43200)

    t = [
        0.008333333333333333,
        0.0,
        -180.0,
        0.0,
        -0.008333333333333333,
        90.0,
        0.0,
        0.0,
        1.0,
    ]
    store = zarr.storage.MemoryStore(root="hazard.zarr")
    root = zarr.open(store=store, mode="w")

    for path, parameter in path_parameters.items():
        add_curves(
            root, longitudes, latitudes, path, shape, parameter, return_periods, t
        )

    return store


def mock_hazard_model_store_single_curve_for_paths(longitudes, latitudes, curve, paths):
    """Create a MemoryStore for creation of Zarr hazard model to be used with unit tests,
    with the specified longitudes and latitudes set to the curve supplied."""

    return_periods = (
        [0.0]
        if len(curve) == 1
        else [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    )
    if len(curve) != len(return_periods):
        raise ValueError(
            f"curve must be single value or of length {len(return_periods)}"
        )

    shape, t = shape_transform_21600_43200(return_periods=return_periods)
    store, root = zarr_memory_store()

    for path in paths():
        add_curves(root, longitudes, latitudes, path, shape, curve, return_periods, t)

    return store


def inundation_return_periods():
    return [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]


def mock_hazard_model_store_path_curves(
    longitudes, latitudes, path_curves: Dict[str, np.ndarray]
):
    """Create a MemoryStore for creation of Zarr hazard model to be used with unit tests,
    with the specified longitudes and latitudes set to the curve supplied."""

    t = [
        0.008333333333333333,
        0.0,
        -180.0,
        0.0,
        -0.008333333333333333,
        90.0,
        0.0,
        0.0,
        1.0,
    ]
    store = zarr.storage.MemoryStore(root="hazard.zarr")
    root = zarr.open(store=store, mode="w")

    for path, curve in path_curves.items():
        if len(curve) == 1:
            return_periods = [0.0]
            shape = (1, 21600, 43200)
        else:
            return_periods = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
            if len(curve) != len(return_periods):
                raise ValueError(
                    f"curve must be single value or of length {len(return_periods)}"
                )
            shape = (len(return_periods), 21600, 43200)

        add_curves(root, longitudes, latitudes, path, shape, curve, return_periods, t)

    return store


def degree_day_heat_parameter_set():
    paths = []

    for model, scenario, year in [
        ("mean_degree_days/above/32c/ACCESS-CM2", "historical", 2005),  # 2005
        ("mean_degree_days/below/32c/ACCESS-CM2", "historical", 2005),
        ("mean_degree_days/above/32c/ACCESS-CM2", "ssp585", 2050),
        ("mean_degree_days/below/32c/ACCESS-CM2", "ssp585", 2050),
    ]:
        paths.append(
            get_source_path_osc_chronic_heat(model=model, scenario=scenario, year=year)
        )
    parameters = [300, 300, 600, -200]
    return dict(zip(paths, parameters, strict=False))


def wbgt_gzn_joint_parameter_set():
    paths = []
    # Getting paths for both hazards.
    for model, scenario, year in [
        ("mean_degree_days/above/32c/ACCESS-CM2", "historical", 2005),  # 2005
        ("mean_degree_days/below/32c/ACCESS-CM2", "historical", 2005),
        ("mean_degree_days/above/32c/ACCESS-CM2", "ssp585", 2050),
        ("mean_degree_days/below/32c/ACCESS-CM2", "ssp585", 2050),
    ]:
        paths.append(
            get_source_path_osc_chronic_heat(model=model, scenario=scenario, year=year)
        )
    for model, scenario, year in [
        ("mean_work_loss/high/ACCESS-CM2", "historical", 2005),  # 2005
        ("mean_work_loss/medium/ACCESS-CM2", "historical", 2005),
        ("mean_work_loss/high/ACCESS-CM2", "ssp585", 2050),
        ("mean_work_loss/medium/ACCESS-CM2", "ssp585", 2050),
    ]:
        paths.append(
            get_source_path_osc_chronic_heat(model=model, scenario=scenario, year=year)
        )
    parameters = [300, 300, 600, -200, 0.05, 0.003, 0.11, 0.013]
    return dict(zip(paths, parameters, strict=False))


def inundation_paths():
    paths = []
    for model, scenario, year in [
        ("MIROC-ESM-CHEM", "rcp4p5", 2030),
        ("MIROC-ESM-CHEM", "rcp8p5", 2080),
    ]:
        paths.append(
            get_source_path_wri_riverine_inundation(
                model=model, scenario=scenario, year=year
            )
        )
    for model, scenario, year in [
        ("wtsub/95", "rcp8p5", "2080"),
        ("wtsub/95", "rcp4p5", "2030"),
        ("wtsub", "historical", "hist"),
        ("nosub", "historical", "hist"),
    ]:
        paths.append(
            get_source_path_wri_coastal_inundation(
                model=model, scenario=scenario, year=year
            )
        )
    return paths


def _wri_inundation_prefix():
    return "inundation/wri/v2"


_percentiles_map = {"95": "0", "5": "0_perc_05", "50": "0_perc_50"}
_subsidence_set = {"wtsub", "nosub"}


def get_source_path_wri_coastal_inundation(*, model: str, scenario: str, year: str):
    type = "coast"
    # model is expected to be of the form subsidence/percentile, e.g. wtsub/95
    # if percentile is omitted then 95th percentile is used
    model_components = model.split("/")
    sub = model_components[0]
    if sub not in _subsidence_set:
        raise ValueError(
            "expected model input of the form {subsidence/percentile}, e.g. wtsub/95, nosub/5, wtsub/50"
        )
    perc = "95" if len(model_components) == 1 else model_components[1]
    return os.path.join(
        _wri_inundation_prefix(),
        f"inun{type}_{cmip6_scenario_to_rcp(scenario)}_{sub}_{year}_{_percentiles_map[perc]}",
    )


def get_source_path_wri_riverine_inundation(*, model: str, scenario: str, year: int):
    type = "river"
    return os.path.join(
        _wri_inundation_prefix(),
        f"inun{type}_{cmip6_scenario_to_rcp(scenario)}_{model}_{year}",
    )


def _osc_chronic_heat_prefix():
    return "chronic_heat/osc/v2"


def get_source_path_osc_chronic_heat(*, model: str, scenario: str, year: int):
    type, *levels = model.split("/")

    if type == "mean_degree_days":
        assert levels[0] in ["above", "below"]  # above or below
        assert levels[1] in ["18c", "32c"]  # threshold temperature
        assert levels[2] in ["ACCESS-CM2"]  # gcms
        return (
            _osc_chronic_heat_prefix()
            + "/"
            + f"{type}_v2_{levels[0]}_{levels[1]}_{levels[2]}_{scenario}_{year}"
        )

    elif type == "mean_work_loss":
        assert levels[0] in ["low", "medium", "high"]  # work intensity
        assert levels[1] in ["ACCESS-CM2"]  # gcms
        return (
            _osc_chronic_heat_prefix()
            + "/"
            + f"{type}_{levels[0]}_{levels[1]}_{scenario}_{year}"
        )

    else:
        raise ValueError("valid types are {valid_types}")
