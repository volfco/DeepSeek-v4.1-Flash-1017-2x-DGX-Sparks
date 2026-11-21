#!/usr/bin/env bash
# SASS evidence for 0202, 0210 and 0212, on the head: per-kernel barrier, shared-memory and shuffle counts plus
# ptxas' register/spill usage, old tree against the 0200-0212 tree. Usage: bash sass.sh <bin directory on the head>
set -euo pipefail
bin=${1:-$HOME/tf-bench/followup/bin}
cuobjdump=/usr/local/cuda/bin/cuobjdump
for v in old new; do
    echo "===== $v"
    for kernel in tf_ds_fused_prompt_idx_key_kernel tf_ds_compress2_norm_kernel tf_ds_idx_fused_kernel; do
        sass=$($cuobjdump -sass -fun "$kernel" "$bin/$v-ops.cubin" 2>/dev/null ||
               $cuobjdump -sass -fun "$kernel" "$bin/$v-torch.cubin" 2>/dev/null || true)
        [[ -n $sass ]] || { echo "$kernel: not in this pair"; continue; }
        printf '%-40s BAR.SYNC=%-3s LDS=%-4s STS=%-4s SHFL=%-4s LDG=%-4s STG=%-4s\n' "$kernel" \
            "$(grep -c 'BAR.SYNC' <<<"$sass" || true)" "$(grep -c 'LDS' <<<"$sass" || true)" \
            "$(grep -c 'STS' <<<"$sass" || true)" "$(grep -c 'SHFL' <<<"$sass" || true)" \
            "$(grep -c 'LDG' <<<"$sass" || true)" "$(grep -c 'STG' <<<"$sass" || true)"
    done
    echo "-- ptxas resource usage"
    $cuobjdump -res-usage "$bin/$v-torch.cubin" | grep -E "Function tf_ds_(fused_prompt_idx_key|fused_idx_key|fused_rope_store|fused_prompt_latent|fused_rope_qd|compress2_norm)_kernel|REG:|STACK|SHARED" | head -40
done
