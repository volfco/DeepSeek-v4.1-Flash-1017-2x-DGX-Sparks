#!/usr/bin/env bash
# Run on the off-pair build host with pinned-source, pinned-fatbins and a fresh work directory.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec python3 -I "$here/../kv/cross-package.py" prune "$@"
