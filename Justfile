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
# flags="K=V K2=V2" exports extra TF_DS_*/TF_SPILL_* settings for the restart (both ranks; e.g.
# `just restart-head flags="TF_DS_ENGRAM_LEAD=1 TF_DS_ENGRAM_ROWCACHE_MB=256"`).
restart-head flags="" offline="1":
    #!/usr/bin/env bash
    set -euo pipefail
    # just passes `name=value` args literally positionally: normalize both forms (`just restart-head "K=V"` and
    # `just restart-head flags="K=V" offline="0"`) by stripping the prefixes and keeping flags first.
    f='{{flags}}'; o='{{offline}}'; f="${f#flags=}"; o="${o#offline=}"
    ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=240 {{head}} \
      "cd {{dir}} && export PATH=\"\$HOME/.local/bin:\$PATH\" HF_HUB_OFFLINE=$o $f && ./start.sh restart"

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

# -- TF_DS_LORA (docs/draft-lora.md): a runtime adapter for the DSpark drafter -----------------------------
# The engine is built only inside zig/Dockerfile, so these drive a dev checkout of the fork instead: a TensorFold
# tree at TF_REF with zig/patches 0001-0118 applied. Override per run: just engine-test engine=../tfref-lora
# zig=/path/to/zig.
engine := "../tfref-lora"
zig := "/home/colum/.local/zig-0.17.0/zig"

engine-build:
    cd {{engine}} && {{zig}} build

engine-test:
    cd {{engine}} && {{zig}} build test

# Build, the host tests, then the server harness (185 server + 132 engine-seam tests). The one known flake is
# lane_host.test.prefill interleave; rerun if it is the only failure.
engine-smoke:
    cd {{engine}} && {{zig}} build && {{zig}} build test && ZIG={{zig}} bash zig/tests/server/zig_test.sh

# Restart both ranks with TF_DS_LORA=<folder>. start.sh forwards TF_DS_* to both ranks, but the folder must be
# reachable *inside* the containers: put the adapter under the /nfs1 checkpoint volume and pass its container path
# (/hf/...), or add a mount to start.sh first. lora="" restarts without the adapter (the bit-identical arm).
engine-restart-lora lora="":
    #!/usr/bin/env bash
    set -euo pipefail
    ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=240 {{head}} \
      "cd {{dir}} && export PATH=\"\$HOME/.local/bin:\$PATH\" HF_HUB_OFFLINE=1 TF_DS_LORA='{{lora}}' && ./start.sh restart"

# The zstd recompression check (runs/zstd-recompress/): one slice of the rank cache through zstd -19 on the head,
# lossless-verified; under 3% saving the lossless-trellis-recompression idea is closed (docs/benchmark-gaps-and-
# moe-review.md, section 5). file="" finds the cache under ~/.cache/dsv41-tensorfold/rank-cache-*; size is MiB.
zstd-check file="" size="1024":
    ssh -o ServerAliveInterval=30 {{head}} "cd {{dir}} && bash runs/zstd-recompress/check.sh '{{file}}' {{size}}"

# Export the second-pass series (engine branch second-pass-int, 12 patches) into zig/patches/ 0160+, beside the
# kept 0129-0145 (--keep-generated). Regenerating the kept series is not supported: this appends. dry=1 previews.
export-second-pass dry="":
    #!/usr/bin/env bash
    set -euo pipefail
    # base = the tip zig/patches already reaches (432d800 = TF_REF + 0001-0171 and 0182-0186,0188-0190, verified 2026-11-16);
    # the next series is the one remaining odd, numbered 0191+ (0187 unused: plan 0164 exists as T15; 0189 is plan 0166's test)
    args=(--engine /home/colum/Workspace/tfref-next --branch second-pass-int --base 432d800 --first 0191 --keep-generated)
    [[ '{{dry}}' == "" ]] || args+=(--dry-run)
    ./scripts/port-patches.sh "${args[@]}"

# Fused-kernel parity + CUDA-graph timing + SASS bench (tools/fused_bench). Syncs the two CUDA source trees and the
# harness to the head, builds both trees with the engine's flags, runs the byte-parity + timing matrix against the
# pinned kit cubins (every kernel WP1-WP3 changed plus the unfused fallbacks their switches fall back to), then the
# SASS evidence, and pulls the artifacts into $OUT.
# Defaults: OLD = NEW = ~/Workspace/tfref-fused-next (the fused-next branch), every kernel, json wp0.json, OUT
# runs/fused-next-20261121/wp0.
# Override with environment variables (the recipe inherits them):
#   OLD=~/Workspace/tfref-prune-six-tip NEW=~/Workspace/tfref-fused-next KERNEL=idx just fused-bench
#   KERNEL=idx SHAPE="idx R=7 nb=8 width" JSON=one.json just fused-bench
#   PARITY=1 just fused-bench            # byte parity only (no timing, no idle-production wait): the durable check
#   OUT=runs/fused-next-20261121/wp4 PARITY=1 JSON=wp4-parity.json just fused-bench
fused-bench:
    #!/usr/bin/env bash
    set -euo pipefail
    o=${OLD:-/home/colum/Workspace/tfref-fused-next}
    n=${NEW:-$o}
    k=${KERNEL:-all}
    s=${SHAPE:-}
    j=${JSON:-wp0.json}
    out=${OUT:-runs/fused-next-20261121/wp0}
    pflag=""
    if [[ ${PARITY:-0} == 1 ]]; then pflag=--parity-only; fi
    rsync -a tools/fused_bench/ {{head}}:tf-bench/fused-next/tools/
    rsync -a --include='dsv41_ops.cu' --include='dsv41_torch.cu' --exclude='*' "$o/zig/kernels/cuda/" {{head}}:tf-bench/fused-next/src-old/
    rsync -a --include='dsv41_ops.cu' --include='dsv41_torch.cu' --exclude='*' "$n/zig/kernels/cuda/" {{head}}:tf-bench/fused-next/src-new/
    ssh -o BatchMode=yes {{head}} "cd ~/tf-bench/fused-next && bash tools/build.sh src-old src-new bin && \
      python3 tools/bench.py bin bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins --kernel '$k' ${s:+--shape \"$s\"} $pflag --json '$j' 2>&1 | tee bench.log && \
      bash tools/sass.sh bin --json sass.json 2>&1 | tee sass.log"
    mkdir -p "$out"
    rsync -a {{head}}:tf-bench/fused-next/bench.log {{head}}:tf-bench/fused-next/sass.log \
      "{{head}}:tf-bench/fused-next/$j" {{head}}:tf-bench/fused-next/sass.json "$out/"

# Export the fused-next round (0213-0258) from the rebased worktree into zig/patches/ and verify the chain: a clean
# TF_REF with every zig/patches/*.patch applied in filename order must equal the branch. Numbers come from each
# commit's own subject (the round has gaps), so this is not scripts/port-patches.sh: that tool applies hand-written
# patches before exported ones and this recipe interleaves them. dry=1 previews.
export-fnext dry="" engine="~/Workspace/tfref-fnext-rebase" base="c3b3859" branch="HEAD":
    #!/usr/bin/env bash
    set -euo pipefail
    args=(--engine "{{engine}}" --base "{{base}}" --branch "{{branch}}")
    [[ '{{dry}}' == "" ]] || args+=(--dry-run)
    ./scripts/export-fnext-series.sh "${args[@]}"
