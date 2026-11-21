#!/usr/bin/env bash
# SASS evidence for the fused kernels (opcode histogram by width, registers/spills/shared, occupancy), old vs new.
# Usage: sass.sh <bin directory on the head> [--json out.json]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/sass.py" "$@"
