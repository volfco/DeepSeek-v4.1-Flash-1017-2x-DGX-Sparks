# Source this file: it reads the OpenObserve credentials from the operator's own helper at run time and exports
# them as OBS_USER / OBS_PASS. Nothing secret is stored here.
_o2_src="${O2_CRED_FILE:-$HOME/Workspace/homelab/kubernetes2/.o2q.py}"
_o2_pair="$(python3 -I -c '
import re,sys
s=open(sys.argv[1]).read()
m=re.search(r"b64encode\(b\"([^\"]+)\"\)",s)
sys.stdout.write(m.group(1) if m else "")
' "$_o2_src")"
if [ -z "$_o2_pair" ]; then echo "o2env: could not read credentials from $_o2_src" >&2; return 1 2>/dev/null || exit 1; fi
export OBS_USER="${_o2_pair%%:*}"
export OBS_PASS="${_o2_pair#*:}"
export OBS_URL="${OBS_URL:-https://openobserve1.lab.volf.co}"
export OBS_ORG="${OBS_ORG:-default}"
unset _o2_pair _o2_src
