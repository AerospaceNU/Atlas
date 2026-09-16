#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
if command -v uv >/dev/null 2>&1; then
  exec uv run python "$here/acquire.py" "$@"
fi
exec python3 "$here/acquire.py" "$@"
