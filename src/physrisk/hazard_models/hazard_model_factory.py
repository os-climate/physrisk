from collections import defaultdict
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

from physrisk.data.inventory import Inventory
from physrisk.data.hazard_data_provider import SourcePaths
from physrisk.data.image_creator import ImageCreator
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel.hazard_model import (
    HazardDataRequest,
    HazardDataResponse,
    HazardModel,
)
from physrisk.kernel.hazard_model import (
    HazardModelFactory as HazardModelFactoryPhysrisk,
)
from physrisk.kernel.hazards import (
    CoastalInundation,
    PluvialInundation,
    RiverineInundation,
)

from physrisk.hazard_models.credentials_provider import CredentialsProvider
from physrisk.hazard_models.hazard_cache import GeometryH3BasedCache
from physrisk.hazard_models.jba_hazard_model import JBAHazardModel


class HazardModelFactory(HazardModelFactoryPhysrisk):
    def __init__(
        self,
        cache_store: GeometryH3BasedCache,
        credentials: CredentialsProvider,
        inventory: Inventory,
        source_paths: SourcePaths,
        store: Optional[MutableMapping] = None,
        reader: Optional[ZarrReader] = None,
        default_interpolation: str = "floor",
        zarr_max_workers: int = 32,
    ):
        self.source_paths = source_paths
        self.cache_store = cache_store
        self.credentials = credentials
        self.inventory = inventory
        self.store = store
        self.reader = reader
        self.default_interpolation = default_interpolation
        self.zarr_max_workers = zarr_max_workers

    def hazard_model(
        self,
        interpolation: Optional[str] = None,
        provider_max_requests: Dict[str, int] = {},
        interpolate_years: bool = True,
    ):
        return CompositeHazardModel(
            self.cache_store,
            self.credentials,
            self.source_paths,
            store=self.store,
            reader=self.reader,
            interpolation=interpolation
            if interpolation is not None
            else self.default_interpolation,
            provider_max_requests=provider_max_requests,
            interpolate_years=interpolate_years,
            zarr_max_workers=self.zarr_max_workers,
        )

    def image_creator(self):
        return ImageCreator(self.inventory, self.source_paths, self.reader)


class CompositeHazardModel(HazardModel):
    """Hazard Model that combines internal data from S3 and data sourced via API.
    API-based models currently limited to model from JBA Risk Management, but pattern
    would be extended for other APIs."""

    def __init__(
        self,
        cache_store: GeometryH3BasedCache,
        credentials: CredentialsProvider,
        source_paths: SourcePaths,
        store: Optional[MutableMapping] = None,
        reader: Optional[ZarrReader] = None,
        interpolation: str = "floor",
        provider_max_requests: Dict[str, int] = {},
        restrict_coverage: bool = False,
        interpolate_years: bool = False,
        use_jba_coastal: bool = False,
        zarr_max_workers: int = 32,
    ):
        self.credentials = credentials
        self.max_jba_requests = provider_max_requests.get("jba", -1)
        self.jba_hazard_model = (
            JBAHazardModel(
                cache_store,
                credentials,
                max_requests=self.max_jba_requests,
                restrict_coverage=restrict_coverage,
            )
            if not self.credentials.jba_api_disabled() and self.max_jba_requests > -1
            else None
        )
        self.zarr_hazard_model = ZarrHazardModel(
            source_paths=source_paths,
            reader=reader,
            store=store,
            interpolation=interpolation,
            interpolate_years=interpolate_years,
            zarr_max_workers=zarr_max_workers,
        )
        self.use_jba_coastal = use_jba_coastal

    def _zarr_hint_path(self, request: HazardDataRequest):
        """Is there a hint path directing to the ZarrHazardModel?"""
        return (
            request.hint
            and request.hint.path
            and not request.hint.path.startswith("jba_")
        )

    def get_hazard_data(
        self, requests: Sequence[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        requests_by_model: Dict[HazardModel, List[HazardDataRequest]] = defaultdict(
            list
        )
        for request in requests:
            if (
                self.jba_hazard_model
                and (
                    request.hazard_type == RiverineInundation
                    or request.hazard_type == PluvialInundation
                )
                and not self._zarr_hint_path(request)
            ):
                requests_by_model[self.jba_hazard_model].append(request)
            elif (
                self.jba_hazard_model
                and request.hazard_type == CoastalInundation
                and self.use_jba_coastal
                and not self._zarr_hint_path(request)
            ):
                requests_by_model[self.jba_hazard_model].append(request)
            else:
                requests_by_model[self.zarr_hazard_model].append(request)

        responses: Dict[HazardDataRequest, HazardDataResponse] = {}

        for model, reqs in requests_by_model.items():
            events_reponses = model.get_hazard_data(reqs)
            responses.update(events_reponses)

        return responses
