from __future__ import annotations

from typing import ClassVar

from atlas.data.stac import StacApiClient

INPE_SEARCH = "https://data.inpe.br/bdc/stac/v1/search"


class InpeStacClient(StacApiClient):
    search_url: ClassVar[str] = INPE_SEARCH


class Cbers4MuxClient(InpeStacClient):
    """CBERS-4 MUX surface reflectance (Brazil Data Cube)."""

    default_collection: ClassVar[str] = "CB4-MUX-L4-SR-1"


class Amazonia1WfiClient(InpeStacClient):
    """Amazonia-1 WFI surface reflectance (Brazil Data Cube)."""

    default_collection: ClassVar[str] = "AMZ1-WFI-L4-SR-1"
