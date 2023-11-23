from pydantic import BaseModel

from physrisk.api.v1.common import Assets


class ExamplePortfoliosRequest(BaseModel):
    """Example portfolios request."""


class ExamplePortfoliosResponse(BaseModel):
    """Example portfolios response."""

    assets: Assets
