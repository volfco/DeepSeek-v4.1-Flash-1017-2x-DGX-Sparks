#!/bin/sh
# Scan the review directory for the OpenObserve password or its base64 forms. The needle comes from the environment
# at run time (`. scripts/o2env.sh` first); nothing secret is stored here. Exit 0 = clean.
dir="$(cd "$(dirname "$0")/.." && pwd)"
[ -n "$OBS_PASS" ] || { echo "leakcheck: OBS_PASS not set" >&2; exit 2; }
b64="$(printf '%s:%s' "$OBS_USER" "$OBS_PASS" | base64 | tr -d '\n')"
hits="$(grep -rlF -e "$OBS_PASS" -e "$b64" "$dir" 2>/dev/null)"
if [ -n "$hits" ]; then echo "LEAK in:"; echo "$hits"; exit 1; fi
echo "leakcheck: clean ($(find "$dir" -type f | wc -l) files scanned)"
