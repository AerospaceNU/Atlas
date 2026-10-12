"""Name a missing search credential without printing its value."""

from __future__ import annotations

import os
import re
from collections.abc import Collection

from atlas.agent.layout import (
    EARTHDATA_TOKEN,
    FIRMS_MAP_KEY,
    KEY_VARIABLES,
    resolve_credential,
)

CDSE_USERNAME = "CDSE_USERNAME"
CDSE_PASSWORD = "CDSE_PASSWORD"
CDSE_CREDENTIALS: tuple[str, ...] = (CDSE_USERNAME, CDSE_PASSWORD)
EARTHDATA_CREDENTIALS: tuple[str, ...] = (EARTHDATA_TOKEN,)
FIRMS_CREDENTIALS: tuple[str, ...] = (FIRMS_MAP_KEY,)

_AUTH_STATUSES = frozenset({401, 403})
# A value must never be interpolated into the error, so only env-var names pass.
_VARIABLE_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


class MissingCredentialError(RuntimeError):
    """A source cannot search because a credential variable is unset.

    ``names`` lists the missing variables. The message names those variables
    and does not include a value. Aggregate records this as the source error.
    """

    def __init__(self, *names: str) -> None:
        cleaned = tuple(name for name in names if _VARIABLE_NAME.fullmatch(name))
        self.names = cleaned
        super().__init__(_missing_message(cleaned))


def credential_is_set(name: str) -> bool:
    """True when ``name`` has a non-blank session, environment, or user value.

    ``EARTHDATA_TOKEN`` and ``FIRMS_MAP_KEY`` use the shared credential lookup.
    Other names, including the CDSE pair, are read from the environment.
    The value is not returned.
    """
    if name in KEY_VARIABLES:
        return bool((resolve_credential(name) or "").strip())
    value = os.environ.get(name)
    return bool(value and value.strip())


def raise_for_missing_credential(
    response: object,
    names: tuple[str, ...],
    *,
    satisfied: Collection[str] = (),
) -> None:
    """Turn an auth rejection into :class:`MissingCredentialError` when unset.

    Status 401 and 403 are the provider failures that do not name a variable.
    When every name in ``names`` is set, or listed in ``satisfied`` because
    the client was given an explicit value, the provider status is left for
    the caller. Any other status is left unchanged.

    Args:
        response: Search response. Only ``status_code`` is read.
        names: Credential variables this source needs, in message order.
        satisfied: Names already held by the client, even if the environment
            does not have them.

    Raises:
        MissingCredentialError: The provider refused the search and at least
            one required variable is absent. The message names it.
    """
    if not names:
        return
    status = getattr(response, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int) or status not in _AUTH_STATUSES:
        return
    held = set(satisfied)
    missing = tuple(name for name in names if name not in held and not credential_is_set(name))
    if missing:
        raise MissingCredentialError(*missing)


def _missing_message(names: tuple[str, ...]) -> str:
    """Return the source-error text for ``names``.

    One variable reads ``NAME is not set``. The CDSE pair, and any other
    list of two or more, names every missing variable. No value is included.
    """
    if not names:
        return "credential is not set"
    if len(names) == 1:
        return f"{names[0]} is not set"
    return f"{' and '.join(names)} are not set"
