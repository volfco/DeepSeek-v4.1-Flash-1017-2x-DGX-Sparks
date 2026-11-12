# Push this checkout to the Spark head (gx10-eb0e: 10.50.0.10 on its LAN, 100.95.49.21 over tail).
# Override for another box: just sync-head head=user@host dir=DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
head := "colum@100.95.49.21"
dir := "DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks"

# No --delete: the head's own scripts/local.sh stays, and nothing is removed there. Caches and
# spill runs (native-spill-*) are left out; .env is synced (the head's copy was backed up before it was first
# overwritten).
sync-head:
    rsync -a --info=progress2 \
      --exclude=.git/ --exclude=scripts/local.sh \
      --exclude=__pycache__/ --exclude=.pytest_cache/ --exclude='native-spill-*/' \
      --exclude=runs/matrix/ \
      ./ {{head}}:{{dir}}/
    ssh {{head}} 'git -C {{dir}} status --short'

# Restart both ranks on the head (builds the image when the patches changed, so it takes as long as prepare.sh).
# ssh runs a non-login shell here: ~/.local/bin (where the `hf` CLI lives) is not on its PATH and prepare.sh needs
# it. HF_HUB_OFFLINE=1 reads the checkpoint, Engram's tables and the gated ABLIT overlay from the /nfs1 cache, which
# holds them all at their pinned revisions, so the restart needs no Hugging Face token.
restart-head offline="1":
    ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=240 {{head}} \
      'cd {{dir}} && export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE={{offline}} && ./start.sh restart'

stop-head:
    ssh {{head}} 'cd {{dir}} && ./stop.sh'

# The engine cores' host-side pinning (scripts/cores.sh): takes storage IRQs off 15-19 (they land on 5-9) and sets
# every core's governor to performance, on both Sparks, then installs it as a boot unit (/usr/local/sbin/tf-cores,
# tf-cores.service) so a reboot keeps it. `just check-cores` reports instead of changing anything.
harden-cores:
    #!/usr/bin/env bash
    set -euo pipefail
    w=$(ssh -o BatchMode=yes {{head}} 'grep -m1 "^WORKER=" {{dir}}/.env | cut -d= -f2-')
    ssh -o BatchMode=yes {{head}} 'cd {{dir}} && sudo ./scripts/cores.sh apply || true; sudo ./scripts/cores.sh install'
    ssh -o BatchMode=yes {{head}} "cat {{dir}}/scripts/cores.sh | ssh -o BatchMode=yes $w 'sudo tee /usr/local/sbin/tf-cores >/dev/null && sudo chmod 0755 /usr/local/sbin/tf-cores && sudo bash /usr/local/sbin/tf-cores apply || true; sudo bash /usr/local/sbin/tf-cores install'"

check-cores:
    #!/usr/bin/env bash
    set -euo pipefail
    w=$(ssh -o BatchMode=yes {{head}} 'grep -m1 "^WORKER=" {{dir}}/.env | cut -d= -f2-')
    echo "== head"; ssh -o BatchMode=yes {{head}} 'cd {{dir}} && ./scripts/cores.sh check'
    echo "== worker ($w)"; ssh -o BatchMode=yes {{head}} "cat {{dir}}/scripts/cores.sh | ssh -o BatchMode=yes $w bash -s check"

# The durable, kernel-level half: managed MSI-X IRQs (the NVMe queues) may not be placed on the engine cores at all.
# Edits the ansible-managed grub drop-in and regenerates grub on both Sparks; takes effect at the next reboot (this
# recipe does not reboot). The GLM repo that owns that file (kernel_cmdline_params) needs the same value, or its next
# ansible run reverts it. `just restart-head` after the reboot brings the engine back.
# The to-benchmark.md matrix, signal grade (scripts/bench_matrix.sh): runs on the head, detached (survives ssh
# drops). Poll with matrix-status, stop with matrix-stop, pull the artifacts with matrix-pull. A subset:
# `just matrix-start arms="map dedupe"`.
matrix-start arms="":
    ssh -o ServerAliveInterval=30 {{head}} 'cd {{dir}} && mkdir -p runs/matrix && { nohup ./scripts/bench_matrix.sh {{arms}} > runs/matrix/runner.log 2>&1 < /dev/null & } && echo matrix-started'

matrix-status:
    ssh {{head}} 'cd {{dir}} && tail -n 12 runs/matrix/status.log 2>/dev/null'

matrix-stop:
    ssh {{head}} 'pkill -f "[b]ench_matrix.sh"; pkill -f "[t]ools/bench.py"; pkill -f "[t]ools/validate.py"; pkill -f "[s]tart.sh"; true'

matrix-pull:
    rsync -a --info=progress2 {{head}}:{{dir}}/runs/matrix/ ./runs/matrix/

grub-engine-irq:
    #!/usr/bin/env bash
    set -euo pipefail
    w=$(ssh -o BatchMode=yes {{head}} 'grep -m1 "^WORKER=" {{dir}}/.env | cut -d= -f2-')
    fix='f=/etc/default/grub.d/zzz-spark-ring.cfg; sudo test -f $f; sudo sed -i "s/isolcpus=[^ ]*/isolcpus=managed_irq,0-4,10-14,15-19/" $f; sudo grep -o "isolcpus=[^ ]*" $f; sudo update-grub >/dev/null && echo "grub updated (reboot to apply)"'
    ssh -o BatchMode=yes {{head}} "$fix"
    ssh -o BatchMode=yes {{head}} "ssh -o BatchMode=yes $w '$fix'"

# The same-prompt A/B for TF_DS_ENGRAM_DEDUPE (scripts/dedupe_probe.sh): identical prompts both arms, natural
# prose + the repeated-paragraph ceiling, read counts from the engine histogram. Runs on the head, detached.
dedupe-probe:
    ssh -o ServerAliveInterval=30 {{head}} 'cd {{dir}} && { nohup ./scripts/dedupe_probe.sh > runs/matrix/dedupe-probe.log 2>&1 < /dev/null & } && echo probe-started'
