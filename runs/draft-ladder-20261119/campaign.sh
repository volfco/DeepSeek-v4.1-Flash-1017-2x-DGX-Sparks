#!/usr/bin/env bash
# A sequence of arms on the bench image, then production restored and validated. Detached on the head, recipe root.
#   runs/draft-ladder-20261119/campaign.sh <name> "<label> <exact> [KEY=VALUE ...]" ...
# A file runs/draft-ladder-20261119/STOP ends the sequence before the next arm. A failed arm ends it too. Production
# is restored from .env either way.
set -uo pipefail
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
D=runs/draft-ladder-20261119; name=$1; shift
img=$(cat $D/package/tag.txt)
say() { echo "$(date -u +%FT%TZ) $name $*" | tee -a $D/status.log; }
say "CAMPAIGN START image=$img arms=$#"
for spec in "$@"; do
  [[ -e $D/STOP ]] && { say "STOP file: no more arms"; break; }
  read -r label exact rest <<< "$spec"
  # shellcheck disable=SC2086
  bash $D/run_arm.sh "$label" "$img" "$exact" $rest; rc=$?
  (( rc == 0 )) || { say "arm $label failed rc=$rc: ending the sequence"; break; }
done
say "RESTORE production from .env"
bash $D/arm.sh "restore-$name" production > "$D/restore-$name.arm.log" 2>&1; rc=$?
say "restore rc=$rc $(grep -c 'engram-lead: on' "$D/restore-$name.arm.log") lead banner lines in the arm log"
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$D/restore-$name/validate.json" --compare runs/matrix/base.validate.json > "$D/restore-$name/validate.log" 2>&1; rc=$?
say "production validate rc=$rc: $(tail -1 "$D/restore-$name/validate.log")"
say "CAMPAIGN DONE"
