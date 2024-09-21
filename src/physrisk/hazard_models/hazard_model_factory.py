from collections import defaultdict
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

from physrisk.data.hazard_data_provider import SourcePath
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
from physrisk.kernel.hazards import PluvialInundation, RiverineInundation

from physrisk.hazard_models.credentials_provider import CredentialsProvider
from physrisk.hazard_models.hazard_cache import H3BasedCache
from physrisk.hazard_models.jba_hazard_model import JBAHazardModel


class HazardModelFactory(HazardModelFactoryPhysrisk):
    def __init__(
        self,
        cache_store: H3BasedCache,
        credentials: CredentialsProvider,
        source_paths: Dict[type, SourcePath],
        store: Optional[MutableMapping] = None,
        reader: Optional[ZarrReader] = None,
    ):
        self.source_paths = source_paths
        self.cache_store = cache_store
        self.credentials = credentials
        self.store = store
        self.reader = reader

    def hazard_model(
        self, interpolation: str = "floor", provider_max_requests: Dict[str, int] = {}
    ):
        return CompositeHazardModel(
            self.cache_store,
            self.credentials,
            self.source_paths,
            store=self.store,
            reader=self.reader,
            interpolation=interpolation,
            provider_max_requests=provider_max_requests,
        )


class CompositeHazardModel(HazardModel):
    """Hazard Model that combines internal data from S3 and data sourced via API."""

    def __init__(
        self,
        cache_store: H3BasedCache,
        credentials: CredentialsProvider,
        source_paths: Dict[type, SourcePath],
        store: Optional[MutableMapping] = None,
        reader: Optional[ZarrReader] = None,
        interpolation: str = "floor",
        provider_max_requests: Dict[str, int] = {},
    ):
        self.credentials = credentials
        self.max_jba_requests = provider_max_requests.get("jba", 0)
        self.jba_hazard_model = (
            JBAHazardModel(cache_store, credentials, max_requests=self.max_jba_requests)
            if not self.credentials.jba_api_disabled() and self.max_jba_requests >= 0
            else None
        )
        self.zarr_hazard_model = ZarrHazardModel(
            source_paths=source_paths,
            reader=reader,
            store=store,
            interpolation=interpolation,
        )

    def hazard_model(self, type):
        if self.jba_hazard_model is not None and (
            type == RiverineInundation or type == PluvialInundation
        ):
            return self.jba_hazard_model
        else:
            return self.zarr_hazard_model

    def get_hazard_data(
        self, requests: Sequence[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        requests_by_model: Dict[HazardModel, List[HazardDataRequest]] = defaultdict(
            list
        )

        for request in requests:
            requests_by_model[self.hazard_model(request.hazard_type)].append(request)

        responses: Dict[HazardDataRequest, HazardDataResponse] = {}

        for model, reqs in requests_by_model.items():
            events_reponses = model.get_hazard_data(reqs)
            responses.update(events_reponses)

        return responses
