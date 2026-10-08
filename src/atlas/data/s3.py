from __future__ import annotations

from urllib.parse import urlencode
from xml.etree import ElementTree

import httpx

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


async def list_keys(client: httpx.AsyncClient, bucket: str, prefix: str) -> list[str]:
    """Every object key under ``prefix`` in a public bucket, across all pages.

    Args:
        client: HTTP client used for the anonymous ListObjectsV2 calls.
        bucket: Bucket name, e.g. ``noaa-goes19``.
        prefix: Key prefix to list.

    Returns:
        Keys in S3 order (lexicographic).
    """
    keys: list[str] = []
    for root in await _list_pages(client, bucket, prefix):
        keys.extend(_texts(root, "s3:Contents/s3:Key"))
    return keys


async def list_prefixes(client: httpx.AsyncClient, bucket: str, prefix: str) -> list[str]:
    """Immediate "folders" under ``prefix`` (``CommonPrefixes`` with ``/`` delimiter).

    Args:
        client: HTTP client used for the anonymous ListObjectsV2 calls.
        bucket: Bucket name.
        prefix: Key prefix ending in ``/``.

    Returns:
        Child prefixes, each ending in ``/``.
    """
    prefixes: list[str] = []
    for root in await _list_pages(client, bucket, prefix, delimiter="/"):
        prefixes.extend(_texts(root, "s3:CommonPrefixes/s3:Prefix"))
    return prefixes


async def _list_pages(
    client: httpx.AsyncClient,
    bucket: str,
    prefix: str,
    *,
    delimiter: str | None = None,
) -> list[ElementTree.Element]:
    pages: list[ElementTree.Element] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        resp = await client.get(f"https://{bucket}.s3.amazonaws.com/?{urlencode(params, safe='/')}")
        resp.raise_for_status()
        try:
            root = ElementTree.fromstring(resp.text)
        except ElementTree.ParseError:
            return pages
        pages.append(root)
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=_S3_NS)
        next_token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
        if truncated.lower() != "true" or not next_token or next_token == token:
            return pages
        token = next_token


def _texts(root: ElementTree.Element, path: str) -> list[str]:
    return [node.text for node in root.findall(path, _S3_NS) if node.text]
