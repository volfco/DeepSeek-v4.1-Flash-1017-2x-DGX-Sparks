#!/usr/bin/env bash
# Same-prompt A/B for TF_DS_ENGRAM_DEDUPE (0096): identical prompts in every leg, prompt cache flushed between
# legs (its states persist on disk via the spill tier and hide the Engram work - the first probe attempt measured
# the cache, not the switch; each leg now prints its cached-token total to prove it ran cold). Legs: base, base
# again (cold assertion + same-prompt count variance), dedupe. Probes per leg: a fixed-salt natural-prose prefill
# block and one repeated-paragraph request (the doc's ceiling). Counts come from the engine histogram per probe.
# Usage: ./scripts/dedupe_probe.sh   (on the head; writes runs/matrix/dedupe-probe.txt)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
export PORT="${PORT:-8000}"
URL="http://127.0.0.1:$PORT"
OUT=runs/matrix/dedupe-probe.txt
SPILL_DIR="$(grep -m1 '^SPILL_DIR=' .env | cut -d= -f2-)"
WORKER="$(grep -m1 '^WORKER=' .env | cut -d= -f2-)"
mkdir -p runs/matrix
: > "$OUT"

reads() { curl -s "$URL/metrics" | awk '/^tensorfold:engram_lookup_seconds_count/{print $2}'; }

# The prompt cache's states are the spill tier's points/blocks; clearing both ranks' trees makes every leg cold.
flush_cache() {
  ./stop.sh >>"$OUT" 2>&1
  rm -rf "$SPILL_DIR"/native-spill-*/rank*
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" 'rm -rf /home/colum/.cache/dsv41-spill/native-spill-*/rank*' >>"$OUT" 2>&1 || true
}

probe() { # $1 = leg label
  local b e cached
  b=$(reads)
  python3 -I tools/bench.py --url "$URL" --suite prefill --prefill-sizes 8000,32000 --reps 3 --salt samedupe1 \
    --label "probe-$1" --out "runs/matrix/probe-$1.json" >>"$OUT" 2>&1
  e=$(reads)
  cached=$(python3 -c "import json; print(sum(r.get('cached_tokens', 0) for r in json.load(open('runs/matrix/probe-$1.json'))['runs']))")
  echo "$1 prose: reads $((e - b)) cached_tokens $cached" | tee -a "$OUT"
  b=$(reads)
  python3 - "$URL" <<'EOF' >>"$OUT" 2>&1
import json, sys, urllib.request
p = "The quick brown fox jumps over the lazy dog near the riverbank while the wind keeps blowing softly. " * 120
body = json.dumps({"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p}],
                   "temperature": 0, "max_tokens": 8}).encode()
req = urllib.request.Request(sys.argv[1] + "/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=180))
print("repeated prompt tokens:", r["usage"]["prompt_tokens"], "usage:", json.dumps(r["usage"].get("prompt_tokens_details", {})))
EOF
  e=$(reads); echo "$1 repeated: reads $((e - b))" | tee -a "$OUT"
}

for leg in "base1|" "base2|" "dedupe|TF_DS_ENGRAM_DEDUPE=1"; do
  name="${leg%%|*}"; flags="${leg#*|}"
  echo "=== probe leg $name flags=[$flags]" | tee -a "$OUT"
  flush_cache
  ( [[ -z "$flags" ]] || eval "export $flags"; ./start.sh restart ) >>"$OUT" 2>&1
  probe "$name"
done
echo "probe done" | tee -a "$OUT"
