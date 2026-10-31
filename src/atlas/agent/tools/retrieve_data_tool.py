import asyncio
from datetime import date
from math import cos, radians
from typing import Any

import httpx  # to download the data files
from pydantic import BaseModel, Field, model_validator

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.cli.download import download_assets
from atlas.cli.estimate import estimate_pull, render_pull_estimate
from atlas.compile import aggregate
from atlas.data.base import BBox, PullRequest, SceneKind
from atlas.data.registry import SOURCES, sources


# pydantic model for the inputs to the tool
class GetDataToolInput(BaseModel):
    data_type: SceneKind = Field(
        description=(
            "The kind of satellite data to retrieve. "
            "landcover: prebuilt maps that label each pixel with a land-cover class. "
            "sar: synthetic aperture radar; shows texture, structure, and moisture. "
            "optical: reflected sunlight in visible and infrared bands, like a photograph. "
            "detection: derived event products, such as active-fire hotspots and burnt areas. "
            "dem: ground surface elevation. "
            "landsurface: modeled soil moisture, evaporation, runoff, and snow cover. "
            "lidar: laser height measurements at points along tracks, not full images. "
            "browse: preview images only, not full-resolution data. "
            "thermal: surface temperature from emitted heat. "
            "precipitation: rainfall rates and totals. "
            "altimetry: radar height of sea, lake, and ice surfaces along tracks. "
            "atmosphere: gases in the air, such as NO2, methane, and carbon monoxide."
        )
    )
    start_date: date = Field(
        description=(
            "First day of the search window, YYYY-MM-DD in UTC, inclusive. A satellite passes "
            "over a given spot only every few days, so for optical data start at least 7 days "
            "before the date of interest. Elevation and most land-cover sources ignore "
            "the dates, but they are still required."
        )
    )
    end_date: date = Field(
        description=(
            "Last day of the search window, YYYY-MM-DD in UTC, inclusive. Must be on or after "
            "start_date. For optical data end at least 7 days after the date of interest."
        )
    )
    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
        description=(
            "Latitude of the point of interest in decimal degrees, north positive. Give it "
            "together with longitude, or use bbox instead."
        ),
    )
    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
        description=(
            "Longitude of the point of interest in decimal degrees, east positive."
            "Give it together with latitude, or use bbox instead."
        ),
    )
    radius_km: float = Field(
        default=2.0,
        gt=0.0,
        le=100.0,
        description=(
            "Half-width in kilometers of the square searched around latitude/longitude. "
            "Defaults to 2 km, capped at 100 km. Ignored when bbox is given. Coarse data needs "
            "a larger radius to cover more than a few pixels."
        ),
    )
    bbox: BBox | None = Field(
        default=None,
        description=(
            "Area to search as west, south, east, north in decimal degrees. Use it instead of "
            "latitude/longitude for a region such as a city or a fire perimeter."
        ),
    )
    sources: list[str] | None = Field(
        default=None,
        description=(
            "Registry keys of the sources to search, such as sentinel2 or landsat9. Defaults "
            "to every source of data_type, which is dozens for optical, so name one or two "
            "when you know which you want."
        ),
    )
    max_cloud_cover: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description=(
            "Maximum scene cloud cover as a percentage from 0 to 100. Only meaningful for "
            "optical data."
        ),
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=30,
        description="Maximum number of scenes per source, from 1 to 30. Defaults to 3.",
    )
    download: bool = Field(
        default=False,
        description=(
            "Whether to download the scenes found. Leave it false on the first call: that "
            "call only searches and returns a download size estimate to show the user. Set "
            "it true only after the user confirms the download."
        ),
    )

    @model_validator(mode="after")
    def check_location(self) -> "GetDataToolInput":
        """Checks to make sure that a location has been included, and not both (i.e. lat/long OR BBox, and not both). Returns a value error if so."""
        if (self.latitude is not None and self.longitude is not None) and self.bbox:
            raise ValueError(
                "Cannot include both latitude and longitude coordinates and bounding box coordinates. Please include one or the other, not both."
            )
        elif not (self.latitude is not None and self.longitude is not None) and not self.bbox:
            raise ValueError(
                "Must include values for either a latitude and longitude corrdinates, or bounding box coordinates. Cannot include nothing."
            )
        else:
            return self

    def lat_long_to_bbox(self) -> BBox:
        """Converts a latitude + longitude point + radius_km into bounding box coordinates."""
        if self.bbox:
            return self.bbox  # we already are using the bounding box, so return it

        # check_location guarantees both are set when there is no bbox; this tells mypy
        assert self.latitude is not None and self.longitude is not None

        lat_padding = (
            self.radius_km / 111.32
        )  # one degree of latitude is approximately 111.32 km, so we can use that to convert the radius in km to degrees of latitude
        if abs(self.latitude) != 90.0:  # to avoid division by zero
            long_padding = lat_padding / cos(radians(self.latitude))
        else:
            long_padding = 360.0

        # build a bounding box from the coordinates + paddings
        return BBox(
            west=max(-180.0, self.longitude - long_padding),
            south=max(-90.0, self.latitude - lat_padding),
            east=min(180.0, self.longitude + long_padding),
            north=min(90.0, self.latitude + lat_padding),
        )


class GetDataTool(Tool):
    name = "get_satellite_data"
    description = (
        "Search satellite catalogs for one kind of data over a place and a date range. Give "
        "the place as latitude/longitude with a radius, or as a bbox. By default this only "
        "searches: it returns the scenes found in each source and a download size estimate. "
        "Show that estimate to the user, and call again with download=true only after they "
        "confirm. Downloaded files are saved under data/<source>/ in the workspace and "
        "returned as artifacts."
    )
    trust = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return GetDataToolInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = GetDataToolInput.model_validate(
            arguments
        )  # check to make sure the input is in the valid format

        # make the pull request that gets sent to the satellite data sources
        pull_request = PullRequest(
            start_date=request.start_date,
            end_date=request.end_date,
            bbox=request.lat_long_to_bbox(),
            max_cloud_cover=request.max_cloud_cover,
            limit=request.limit,
        )

        # get the list of sources to search
        if request.sources:
            unknown = [
                name for name in request.sources if name not in SOURCES
            ]  # identify any unknown sources
            if unknown:
                raise ValueError(f"Unknown source(s): {unknown}. Known: {sorted(SOURCES)}")
            names = list(request.sources)
        else:
            names = [
                name for name, cls in SOURCES.items() if cls.scene_kind == request.data_type
            ]  # get all sources that match the requested data type

        # safety checks
        if not names:
            raise ValueError(f"No sources for data type {request.data_type.value}")
        if "fldas" in names:
            raise ValueError(
                "fldas builds a CSV table, not files. Use `atlas data fldas landsurface` instead."
            )

        # _pull() can take time to download the data, and so we need to make it async
        return asyncio.run(self._pull(pull_request, names, store, download=request.download))

    async def _pull(
        self,
        pull_request: PullRequest,
        names: list[str],
        store: LocalArtifactStore,
        *,
        download: bool,
    ) -> ToolResult:
        """Pull data from the specified sources based on the pull request.

        Args:
            pull_request (PullRequest): The pull request containing the search criteria.
            names (list[str]): The list of source names to pull data from.
            store (LocalArtifactStore): The local artifact store where data will be saved.
            download (bool): Whether to download the data or just estimate it.

        Returns:
            ToolResult: The result of the pull operation, including either the download report or the search estimate.
        """
        clients = sources(*names)
        out_dir = store.resolve("data")  # where to put the data
        try:
            catalog = await aggregate(pull_request, clients)

            if not download:
                # this case is solely if we are searching for data and we dont want to download it (since downloading takes time and storage)
                estimate = estimate_pull(
                    catalog, out_dir=out_dir, bbox=pull_request.bbox
                )  # estimate the scenes and the bytes for the download
                lines = [render_pull_estimate(estimate), "scenes:"]

                for src in catalog.sources:  # for all sources that we pull data from
                    for scene in src.scenes:  # for all scenes from this source
                        # get the cloud cover for this scene, or n/a if the catalog has none
                        cloud = "n/a" if scene.cloud_cover is None else f"{scene.cloud_cover:.0f}%"

                        # make a line describing the scene with its source, ID, date, and cloud cover
                        lines.append(
                            f"  {src.source} {scene.id} {scene.datetime:%Y-%m-%d} cloud {cloud}"
                        )

                return ToolResult(
                    text="\n".join(lines)
                )  # return the search results as a ToolResult object

            # download every scene from sources whose search worked
            scenes = []  # this will hold the scenes that we intend to download
            for src in catalog.sources:
                if src.error is None:  # skip sources whose search failed, since they have no scenes
                    scenes.extend(
                        src.scenes
                    )  # add all of this source's scenes to the download list

            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as http:
                # download one file per scene into out_dir/<source>/
                report = await download_assets(
                    scenes, out_dir, http=http, clients=clients, bbox=pull_request.bbox
                )

        finally:
            for client in clients.values():  # for every client we created
                close = getattr(client, "aclose", None)
                if close is not None:  # if the client has an aclose method
                    await close()  # close this client's network connections

        artifacts = [
            store.relative(path) for paths in report.saved_paths().values() for path in paths
        ]

        lines = []  # the lines of text that go back to the agent
        for src in catalog.sources:  # one summary line per source
            if src.error is not None:
                # this source's search failed, so report the error and move on to the next source
                lines.append(f"{src.source}: search failed: {src.error}")
                continue

            outcomes = report.for_source(src.source)
            # count the scenes that are on disk now. skipped means it was already downloaded earlier
            saved = sum(1 for o in outcomes if o.status in {"saved", "skipped"})

            lines.append(f"{src.source}: {saved} of {len(src.scenes)} scenes saved")
            # add an indented line for every scene that had an error, with its ID and the error
            lines.extend(f"  {o.scene_id}: {o.error}" for o in outcomes if o.error)

        lines.extend(f"saved: {path}" for path in artifacts)  # list every saved file at the end

        # join the lines into one string and return it with the list of saved files
        return ToolResult(text="\n".join(lines), artifacts=artifacts)
