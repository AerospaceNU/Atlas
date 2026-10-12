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
    and says how to set them. It does not include a value. Aggregate records
    this as the source error.
    """

    def __init__(self, *names: str) -> None:
        cleaned = tuple(name for name in names if _VARIABLE_NAME.fullmatch(name))
        self.names = cleaned
        super().__init__(_missing_message(cleaned))


class CredentialRejectedError(RuntimeError):
    """A provider returned 401 or 403 while the credential was present.

    ``names`` lists the variables that were sent or already held. The message
    names those variables and the HTTP status. It does not include a value.
    """

    def __init__(self, status: int, *names: str) -> None:
        cleaned = tuple(name for name in names if _VARIABLE_NAME.fullmatch(name))
        self.names = cleaned
        self.status_code = status
        super().__init__(_rejected_message(status, cleaned))


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
    """Turn an auth rejection into a named credential error.

    Status 401 and 403 are the provider failures that do not name a variable.
    A name that is unset, and not listed in ``satisfied``, becomes
    :class:`MissingCredentialError`. When every name is set, or listed in
    ``satisfied`` because the client was given an explicit value, the error
    is :class:`CredentialRejectedError`. Any other status is left unchanged.

    Args:
        response: Search response. Only ``status_code`` is read.
        names: Credential variables this source needs, in message order.
        satisfied: Names already held by the client, even if the environment
            does not have them.

    Raises:
        MissingCredentialError: The provider refused the search and at least
            one required variable is absent. The message names it and says
            how to set it.
        CredentialRejectedError: The provider refused the search and every
            required variable is present. The message names it and the status.
            The value is not included.
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
    raise CredentialRejectedError(status, *names)


# ``/key`` stores these two. CDSE has no slash command, so its hint is the env var.
_KEY_COMMAND = {
    EARTHDATA_TOKEN: "earthdata",
    FIRMS_MAP_KEY: "firms",
}


def _missing_message(names: tuple[str, ...]) -> str:
    """Return the source-error text for ``names``.

    One variable reads ``NAME is not set`` plus how to set it. Earthdata and
    FIRMS name ``/key`` and the env var. CDSE, and any other name, names the
    env var only. Two or more variables are joined the same way. No value is
    included.
    """
    if not names:
        return "credential is not set"
    hint = _set_hint(names)
    if len(names) == 1:
        return f"{names[0]} is not set; {hint}"
    return f"{' and '.join(names)} are not set; {hint}"


def _set_hint(names: tuple[str, ...]) -> str:
    """Return the clause that says how to set ``names``."""
    commands = tuple(_KEY_COMMAND[name] for name in names if name in _KEY_COMMAND)
    if len(commands) == len(names):
        if len(names) == 1:
            return f"set it with /key {commands[0]} or the env var"
        listed = " or ".join(f"/key {command}" for command in commands)
        return f"set them with {listed} or the env vars"
    if len(names) == 1:
        return "set the env var"
    return "set the env vars"


def _rejected_message(status: int, names: tuple[str, ...]) -> str:
    """Return the source-error text for a present credential the provider refused.

    One variable reads ``NAME was rejected (HTTP status)``. Two or more use
    ``were``. No value is included.
    """
    if not names:
        return f"credential was rejected (HTTP {status})"
    if len(names) == 1:
        return f"{names[0]} was rejected (HTTP {status})"
    return f"{' and '.join(names)} were rejected (HTTP {status})"
