from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar, NamedTuple, Self
from urllib.parse import parse_qsl, urljoin, urlparse

import httpx

from atlas.data.bands import resolve_bands
from atlas.data.base import (
    Asset,
    BBox,
    DataPullClient,
    GeometryKind,
    PullRequest,
    PullResult,
    Scene,
    SceneKind,
)
from atlas.data.credentials import raise_for_missing_credential

_CLOUD_QUERY_KINDS = frozenset({SceneKind.optical, SceneKind.browse})


class StacApiClient(DataPullClient):
    search_url: ClassVar[str] = ""
    default_collection: ClassVar[str] = ""
    include_datetime: ClassVar[bool] = True
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = None
    # Unset variables named here replace a 401 or 403 with MissingCredentialError.
    required_credentials: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        collection: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not self.search_url:
            raise ValueError(f"{type(self).__name__} has no search_url")
        if collection is None and not self.default_collection:
            raise ValueError(
                f"{type(self).__name__} has no default_collection; pass `collection=...`"
            )
        self.collection = collection or self.default_collection
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def search(self, request: PullRequest) -> PullResult:
        resp = await self._client.post(self.search_url, json=self._search_body(request))
        self._raise_for_credentials(resp)
        resp.raise_for_status()
        return self._result_from_payload(request, resp.json())

    async def search_from_cursor(self, request: PullRequest, page_cursor: str) -> PullResult:
        """Request the STAC page named by ``page_cursor``.

        A JSON cursor carries the next link's method, href, merge flag, and
        body. ``merge`` true updates the original search body with that body.
        Otherwise the link body is the whole POST body. A legacy pagination
        token is posted with the original search body. A legacy http(s) URL on
        this client's search host is fetched with GET when its query is not
        signed. The page-one request is not sent.

        Args:
            request: Pull whose filters produced ``page_cursor``.
            page_cursor: JSON next link, legacy token, or same-host next URL.

        Returns:
            That page, including the following cursor when the response has one.

        Raises:
            ValueError: The cursor is on another host, carries a signed query,
                or is not a usable next link. The message does not include the URL.
            MissingCredentialError: The provider returned 401 or 403 and a
                variable in ``required_credentials`` is unset. The message
                names that variable, says how to set it, and does not include
                a value.
            CredentialRejectedError: The provider returned 401 or 403 and
                every required variable is set. The message names the
                variable and the status, not the value.
        """
        call = _cursor_request(page_cursor, self.search_url, request, self._search_body)
        if call.method == "GET":
            resp = await self._client.get(call.url)
        else:
            resp = await self._client.post(call.url, json=call.body)
        self._raise_for_credentials(resp)
        resp.raise_for_status()
        return self._result_from_payload(request, resp.json())

    def _raise_for_credentials(self, response: object) -> None:
        """Raise a named credential error for a 401 or 403.

        An unset variable is missing. A variable that is set was rejected.
        Any other status is left for the caller. The value is not included.
        """
        raise_for_missing_credential(response, self.required_credentials)

    def _search_body(self, request: PullRequest, *, token: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "collections": [self.collection],
            "bbox": request.bbox.as_list(),
            "limit": request.limit,
        }
        if request.polygon is not None:
            body["intersects"] = request.polygon.as_stac_geometry()
        if self.include_datetime:
            body["datetime"] = (
                f"{request.start_date.isoformat()}T00:00:00Z/"
                f"{request.end_date.isoformat()}T23:59:59Z"
            )
        query = self._build_query_filter(request)
        if query:
            body["query"] = query
        if token:
            body["token"] = token
        return body

    def _result_from_payload(self, request: PullRequest, payload: Any) -> PullResult:
        if not isinstance(payload, dict):
            raise TypeError(f"Expected STAC FeatureCollection, got {type(payload)}")
        scenes: list[Scene] = []
        for feature in payload.get("features") or []:
            if not isinstance(feature, dict):
                continue
            scene = item_to_scene(
                feature,
                default_platform=self.collection,
                collection=self.collection,
                kind=self.scene_kind,
                gsd_m=self.nominal_gsd_m,
            )
            if scene is not None:
                scenes.append(scene)
        cursor, unavailable = _next_page_cursor(payload, self.search_url)
        return PullResult(
            request=request,
            scenes=scenes,
            next_cursor=cursor,
            next_cursor_unavailable=unavailable,
        )

    def _cloud_query(self, request: PullRequest) -> dict[str, Any] | None:
        if self.scene_kind not in _CLOUD_QUERY_KINDS:
            return None
        bounds: dict[str, float] = {}
        if request.min_cloud_cover is not None:
            bounds["gte"] = request.min_cloud_cover
        if request.max_cloud_cover is not None:
            bounds["lte"] = request.max_cloud_cover
        if not bounds:
            return None
        return {"eo:cloud_cover": bounds}

    def _build_query_filter(self, request: PullRequest) -> dict[str, Any] | None:
        return self._cloud_query(request)


# Query keys that grant access. A page token is not one of them.
_SIGNED_QUERY_KEYS = frozenset(
    {
        "sig",
        "signature",
        "skoid",
        "sktid",
        "skt",
        "ske",
        "sks",
        "skv",
        "awsaccesskeyid",
        "securitytoken",
        "sv",
        "ss",
        "srt",
        "sp",
        "se",
        "st",
        "spr",
        "sr",
    }
)


class _CursorCall(NamedTuple):
    method: str
    url: str
    body: dict[str, Any] | None


def encode_page_cursor(fields: Mapping[str, Any]) -> str:
    """Return the canonical JSON cursor, or ``""`` when ``fields`` has nothing to store.

    Key order is fixed so a manifest object and this string describe the same cursor.
    """
    payload: dict[str, Any] = {}
    method = fields.get("method")
    if isinstance(method, str) and method.strip():
        method_text = method.strip().upper()
        if method_text in {"GET", "POST"}:
            payload["method"] = method_text
    href = fields.get("href")
    if isinstance(href, str) and href.strip():
        payload["href"] = href.strip()
    merge = fields.get("merge")
    if isinstance(merge, bool):
        payload["merge"] = merge
    body = fields.get("body")
    if isinstance(body, dict) and body:
        payload["body"] = body
    if not payload:
        return ""
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return ""


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _same_origin(url: str, search_url: str) -> bool:
    left, right = urlparse(url), urlparse(search_url)
    return left.scheme == right.scheme and bool(left.netloc) and left.netloc == right.netloc


def _is_bare_search(url: str, search_url: str) -> bool:
    parsed, search = urlparse(url), urlparse(search_url)
    return parsed.path.rstrip("/") == search.path.rstrip("/") and not parsed.query


def _query_is_signed(url: str) -> bool:
    for key, _value in parse_qsl(urlparse(url).query, keep_blank_values=True):
        lowered = key.lower()
        if (
            lowered in _SIGNED_QUERY_KEYS
            or lowered.startswith("x-amz-")
            or lowered.startswith("x-goog-")
        ):
            return True
    return False


def _next_page_cursor(payload: Mapping[str, Any], search_url: str) -> tuple[str | None, str | None]:
    """Return the next-page cursor from a STAC Item Search response.

    A ``links`` entry with ``rel`` ``next`` is the cursor. Its body is kept
    whole, including keys other than ``token``, along with ``method`` and
    ``merge``. A GET href on the search host is stored when the query is not
    signed. A signed query is not stored. When that signed href is the only
    continuation, the cursor is unavailable with reason ``signed`` rather
    than looking like the last page. One page is read; the cursor is only
    recorded.

    Args:
        payload: STAC FeatureCollection.
        search_url: This client's item-search URL, used to resolve relative links.

    Returns:
        The JSON cursor and None, or None and an unavailable reason. Both are
        None when the response has no next page.
    """
    links = payload.get("links")
    if not isinstance(links, list):
        return None, None
    unavailable: str | None = None
    for link in links:
        if not isinstance(link, dict) or link.get("rel") != "next":
            continue
        cursor, reason = _cursor_from_link(link, search_url)
        if cursor:
            return cursor, None
        if reason and unavailable is None:
            unavailable = reason
    return None, unavailable


def _json_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):
        return None
    parsed = json.loads(encoded)
    if not isinstance(parsed, dict) or not parsed:
        return None
    return parsed


def _link_method(link: Mapping[str, Any], body: dict[str, Any] | None) -> str | None:
    raw = link.get("method")
    if isinstance(raw, str) and raw.strip():
        method = raw.strip().upper()
        if method not in {"GET", "POST"}:
            return None
        return method
    if body:
        return "POST"
    return "GET"


def _classify_href(href: object, search_url: str) -> tuple[str | None, str]:
    """Return the absolute href and one of missing, signed, cross-origin, bare, or ok."""
    if not isinstance(href, str) or not href.strip():
        return None, "missing"
    absolute = urljoin(search_url, href.strip())
    if not _is_http_url(absolute):
        return None, "missing"
    if _query_is_signed(absolute):
        return None, "signed"
    if not _same_origin(absolute, search_url):
        return None, "cross-origin"
    if _is_bare_search(absolute, search_url):
        return absolute, "bare"
    return absolute, "ok"


def _cursor_from_link(link: Mapping[str, Any], search_url: str) -> tuple[str | None, str | None]:
    """Return a stored cursor, or None and ``signed`` when that is the only continuation."""
    body = _json_object(link.get("body"))
    method = _link_method(link, body)
    if method is None:
        return None, None
    href, status = _classify_href(link.get("href"), search_url)
    if method == "GET":
        if status == "ok" and href is not None:
            return encode_page_cursor({"method": "GET", "href": href}) or None, None
        if status == "signed" and body is None:
            return None, "signed"
        return None, None
    stored_href = href if status in {"ok", "bare"} else None
    if body is None and status == "signed":
        return None, "signed"
    if body is None and (stored_href is None or status == "bare"):
        return None, None
    spec: dict[str, Any] = {"method": "POST", "merge": link.get("merge") is True}
    if stored_href is not None:
        spec["href"] = stored_href
    if body is not None:
        spec["body"] = body
    return encode_page_cursor(spec) or None, None


def _cursor_request(
    page_cursor: str,
    search_url: str,
    request: PullRequest,
    search_body: Callable[..., dict[str, Any]],
) -> _CursorCall:
    text = page_cursor.strip()
    if not text:
        raise ValueError("next-page cursor is not valid")
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("next-page cursor is not valid") from exc
        if not isinstance(payload, dict):
            raise ValueError("next-page cursor is not valid")
        return _call_from_object(payload, search_url, request, search_body)
    if _is_http_url(text):
        return _get_call(text, search_url)
    return _CursorCall("POST", search_url, search_body(request, token=text))


def _get_call(url: str, search_url: str) -> _CursorCall:
    if _query_is_signed(url):
        raise ValueError("next-page cursor contains a signed query and was not fetched")
    if not _same_origin(url, search_url):
        raise ValueError("next-page cursor host does not match the STAC search host")
    return _CursorCall("GET", url, None)


def _call_from_object(
    payload: Mapping[str, Any],
    search_url: str,
    request: PullRequest,
    search_body: Callable[..., dict[str, Any]],
) -> _CursorCall:
    link_body = payload.get("body")
    body = link_body if isinstance(link_body, dict) and link_body else None
    raw_method = payload.get("method")
    if isinstance(raw_method, str) and raw_method.strip():
        method = raw_method.strip().upper()
        if method not in {"GET", "POST"}:
            raise ValueError("next-page cursor method is not supported")
    elif body:
        method = "POST"
    else:
        method = "GET"
    href = payload.get("href")
    url = search_url
    if isinstance(href, str) and href.strip():
        absolute = urljoin(search_url, href.strip())
        if _query_is_signed(absolute):
            raise ValueError("next-page cursor contains a signed query and was not fetched")
        if not _same_origin(absolute, search_url):
            raise ValueError("next-page cursor host does not match the STAC search host")
        url = absolute
    elif method == "GET":
        raise ValueError("next-page cursor is not valid")
    if method == "GET":
        return _CursorCall("GET", url, None)
    if payload.get("merge") is True:
        posted = search_body(request)
        if body:
            posted.update(body)
        return _CursorCall("POST", url, posted)
    if body is None:
        return _CursorCall("POST", url, {})
    return _CursorCall("POST", url, dict(body))


def _raster_calibration(raw: object) -> tuple[float | None, float | None]:
    """Return scale and offset from the first ``raster:bands`` entry."""
    if not isinstance(raw, list) or not raw:
        return None, None
    entry = raw[0]
    if not isinstance(entry, dict):
        return None, None
    return _finite_float(entry.get("scale")), _finite_float(entry.get("offset"))


def _finite_float(raw: object) -> float | None:
    """Return a finite float, or None for anything else."""
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def item_to_scene(
    feature: dict[str, Any],
    *,
    default_platform: str = "unknown",
    source: str = "",
    collection: str = "",
    kind: SceneKind = SceneKind.optical,
    gsd_m: float | None = None,
) -> Scene | None:
    scene_id = feature.get("id")
    if not isinstance(scene_id, str) or not scene_id:
        return None
    props = feature.get("properties")
    if not isinstance(props, dict):
        props = {}
    scene_dt = _stac_datetime(props.get("datetime") or props.get("start_datetime"))
    if scene_dt is None:
        return None

    parsed = _parse_geometry(feature)
    if parsed is None:
        return None
    bbox, geometry_kind, lon, lat = parsed

    assets: dict[str, Asset] = {}
    band_aliases: dict[str, str] = {}
    raw_assets = feature.get("assets")
    if isinstance(raw_assets, dict):
        for name, spec in raw_assets.items():
            if not isinstance(spec, dict) or not isinstance(spec.get("href"), str):
                continue
            roles = spec.get("roles")
            scale, offset = _raster_calibration(spec.get("raster:bands"))
            asset = Asset.try_new(
                href=_http_href(spec["href"]),
                media_type=spec.get("type") if isinstance(spec.get("type"), str) else None,
                title=spec.get("title") if isinstance(spec.get("title"), str) else None,
                roles=list(roles) if isinstance(roles, list) else [],
                scale=scale,
                offset=offset,
            )
            if asset is not None:
                assets[str(name)] = asset
        # `eo:bands` lives only on the raw specs, so resolve before the loop
        # above drops it -- but only over assets the loop kept, or a rejected
        # one could win the match and shadow a usable band.
        band_aliases = resolve_bands(
            {name: spec for name, spec in raw_assets.items() if str(name) in assets}
        )

    instruments = props.get("instruments")
    instrument: str | None = None
    if isinstance(instruments, list) and instruments and isinstance(instruments[0], str):
        instrument = instruments[0]

    item_gsd = props.get("gsd")
    resolved_gsd = (
        float(item_gsd)
        if isinstance(item_gsd, int | float) and not isinstance(item_gsd, bool)
        else gsd_m
    )

    return Scene.try_new(
        id=scene_id,
        source=source,
        collection=collection,
        kind=kind,
        datetime=scene_dt,
        start_datetime=_stac_datetime(props.get("start_datetime")),
        end_datetime=_stac_datetime(props.get("end_datetime")),
        bbox=bbox,
        geometry_kind=geometry_kind,
        lon=lon,
        lat=lat,
        gsd_m=resolved_gsd,
        platform=_stac_platform(props, default_platform),
        instrument=instrument,
        cloud_cover=_stac_cloud_cover(props.get("eo:cloud_cover")),
        assets=assets,
        band_aliases=band_aliases,
        properties=props,
    )


def _parse_geometry(
    feature: dict[str, Any],
) -> tuple[BBox, GeometryKind, float | None, float | None] | None:
    geom = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else None
    coords = geom.get("coordinates") if isinstance(geom, dict) else None
    geom_type = geom.get("type") if isinstance(geom, dict) else None

    bbox = _stac_bbox(feature)
    if bbox is not None:
        if geom_type == "Point" and isinstance(coords, list) and len(coords) >= 2:
            try:
                return bbox, GeometryKind.point, float(coords[0]), float(coords[1])
            except (TypeError, ValueError):
                return bbox, GeometryKind.bbox, None, None
        if geom_type in {"Polygon", "MultiPolygon"}:
            return bbox, GeometryKind.polygon, None, None
        return bbox, GeometryKind.bbox, None, None

    if geom_type == "Point" and isinstance(coords, list) and len(coords) >= 2:
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError, IndexError):
            return None
        bbox = BBox.try_new(
            west=max(-180.0, lon - 0.01),
            south=max(-90.0, lat - 0.01),
            east=min(180.0, lon + 0.01),
            north=min(90.0, lat + 0.01),
        )
        if bbox is None:
            return None
        return bbox, GeometryKind.point, lon, lat
    return None


def _stac_datetime(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _stac_bbox(feature: dict[str, Any]) -> BBox | None:
    bbox_list = feature.get("bbox")
    if not isinstance(bbox_list, list) or len(bbox_list) not in {4, 6}:
        return None
    # A 6-number bbox is [west, south, floor, east, north, ceiling].
    half = len(bbox_list) // 2
    try:
        return BBox.try_new(
            west=float(bbox_list[0]),
            south=float(bbox_list[1]),
            east=float(bbox_list[half]),
            north=float(bbox_list[half + 1]),
        )
    except (TypeError, ValueError):
        return None


def _stac_cloud_cover(raw: object) -> float | None:
    if isinstance(raw, str):
        try:
            value = float(raw)
        except ValueError:
            return None
    elif isinstance(raw, int | float) and not isinstance(raw, bool):
        value = float(raw)
    else:
        return None
    if not math.isfinite(value):
        return None
    return min(max(value, 0.0), 100.0)


def _http_href(href: str) -> str:
    """Expose S3 object URLs over HTTPS so callers can fetch without an s3 client."""
    if not href.startswith("s3://"):
        return href
    rest = href.removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        return href
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def _stac_platform(props: dict[str, Any], default: str) -> str:
    for key in ("const:platform", "platform"):
        raw = props.get(key)
        if isinstance(raw, str) and raw:
            return raw
        if isinstance(raw, list) and raw and isinstance(raw[0], str) and raw[0]:
            return raw[0]
    return default
