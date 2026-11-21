#!/usr/bin/env bash
# Host-side pinning for the engine's cores, one Spark at a time (root: `sudo ./scripts/cores.sh apply`).
#
# start.sh pins both ranks' containers to CPUSET (15-19: the Cortex-X925s of the 16 MB-L3 cluster) and the host's
# docker cgroup parent puts every other container on 10-19, so no user-space task outside tensorfold can run on 15-19.
# Two kernel-side things still reach those cores, and this keeps them off:
#   - IRQs. The NVMe driver pins one queue per CPU (nvme0q16..q20 on the head, q13..q17 on the worker, one each to
#     15..19): measured ~3,900 interrupts/s on CPU 16 while the engine read Engram's tables. Those are managed MSI-X
#     IRQs, and the kernel refuses a userspace affinity write for them (EIO), so `apply` reports them and leaves them
#     alone: `isolcpus=managed_irq` at boot is their fix (the Justfile's grub-engine-irq). A non-managed IRQ found
#     inside the engine set still moves to the helper CPUs (5-9: the X925s of the 8 MB-L3 cluster, in the OS slice).
#     mlx5 (RDMA) queues are left alone: they are not pinned into 15-19 here, and moving them would change the ring's
#     completion path.
#   - Clocks. `scaling_governor=performance` is set on both Sparks, but nothing enforces it across a reboot; `apply`
#     sets it again on every core and the installed unit (see the Justfile's harden-cores) does so at boot.
# Per-CPU kernel threads (ksoftirqd/N, migration/N, kworker/N:0) stay: they are idle at rest and cannot be moved. The
# kernel-level answer to the NVMe queues is `isolcpus=managed_irq,15-19` in the cmdline (the Justfile's grub-engine-irq,
# a reboot), which keeps managed MSI-X IRQs off them even after a driver reload.
#
# Usage: cores.sh [check|apply|install]     ENGINE_CPUS=15-19 HELPER_CPUS=5-9 override the sets.
set -euo pipefail
ENGINE_CPUS="${ENGINE_CPUS:-15-19}"
HELPER_CPUS="${HELPER_CPUS:-5-9}"
MODE="${1:-check}"

expand() { # "15-19,21" -> "15 16 17 18 19 21"
    local part lo hi
    IFS=', ' read -ra parts <<< "$1"
    for part in "${parts[@]}"; do
        case "$part" in
            *-*) lo="${part%-*}"; hi="${part#*-}"; seq "$lo" "$hi" ;;
            *) echo "$part" ;;
        esac
    done
}

engine=" $(expand "$ENGINE_CPUS" | tr '\n' ' ')"
inside_engine() { # every cpu of the list is in the engine set
    local c
    for c in $(expand "$1"); do [[ " $engine " == *" $c "* ]] || return 1; done
    return 0
}

# IRQs whose every CPU is an engine core, by the device line in /proc/interrupts
engine_irqs() {
    local dir n owner
    for dir in /proc/irq/[0-9]*/; do
        n="${dir#/proc/irq/}"; n="${n%/}"
        [[ -r "$dir/smp_affinity_list" ]] || continue
        inside_engine "$(cat "$dir/smp_affinity_list")" || continue
        owner=$(awk -v i="$n:" '$1 == i { print $(NF-2), $(NF-1), $NF }' /proc/interrupts | tail -1)
        printf '%s\t%s\t%s\n' "$n" "$(cat "$dir/smp_affinity_list")" "$owner"
    done
}

governors() {
    local g
    for g in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
        printf '%s %s\n' "$(cat "${g%/scaling_governor}/cpuinfo_max_freq" 2>/dev/null)" "$(cat "$g")"
    done | sort -n | awk '{ c[$1" "$2]++ } END { for (k in c) printf "  governor=%s max_freq=%s cores=%d\n", substr(k, index(k," ")+1), substr(k,1,index(k," ")-1), c[k] }'
}

case "$MODE" in
    check)
        echo "engine cores: $ENGINE_CPUS (containers' --cpuset-cpus), helpers: $HELPER_CPUS"
        echo "governors:"
        governors
        echo "IRQs wholly inside the engine cores:"
        found=$(engine_irqs); [[ -n "$found" ]] || { echo "  none"; exit 0; }
        printf '%s\n' "$found" | while IFS=$'\t' read -r n cpus owner; do
            printf '  irq%-4s cpus=%-6s %s\n' "$n" "$cpus" "$owner"
        done
        count=$(awk -v e="$(expand "$ENGINE_CPUS" | tr '\n' ',')" 'BEGIN{n=0} {n++} END{print n}' <<< "$found")
        echo "  ($count to move: sudo $0 apply)"
        ;;
    apply)
        [[ "$(id -u)" == 0 ]] || { echo "apply needs root: sudo $0 apply" >&2; exit 1; }
        for g in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
            [[ -w "$g" ]] && echo performance > "$g" || true
        done
        moved=0
        skipped=0
        while IFS=$'\t' read -r n cpus owner; do
            [[ -n "$n" ]] || continue
            if echo "$HELPER_CPUS" > "/proc/irq/$n/smp_affinity_list" 2>/dev/null; then
                echo "irq$n ($owner) cpus=$cpus -> $HELPER_CPUS"
                moved=$((moved + 1))
            elif [[ "$owner" == *Edge* || "$owner" == *Level* ]]; then
                # EIO: a managed MSI-X IRQ (the NVMe queue IRQs are) keeps the affinity the driver asked for; only
                # isolcpus=managed_irq at boot keeps them off a set of CPUs (see the Justfile's grub-engine-irq)
                echo "irq$n ($owner): managed IRQ, the kernel owns its affinity (EIO): needs isolcpus=managed_irq at boot"
                skipped=$((skipped + 1))
            else
                echo "irq$n ($owner): the kernel refused $HELPER_CPUS" >&2
            fi
        done <<< "$(engine_irqs)"
        echo "governors now:"; governors
        echo "moved $moved IRQ(s) off the engine cores; $skipped managed IRQ(s) left (the boot arg is their fix)"
        ;;
    install) # keep it across reboots: /usr/local/sbin/tf-cores + tf-cores.service, applied now
        [[ "$(id -u)" == 0 ]] || { echo "install needs root: sudo $0 install" >&2; exit 1; }
        self=$(readlink -f "$0")
        if [[ -f "$self" && "$self" != /usr/local/sbin/tf-cores ]]; then install -m 0755 "$self" /usr/local/sbin/tf-cores; fi
        [[ -x /usr/local/sbin/tf-cores ]] || { echo "no /usr/local/sbin/tf-cores (piped in? write it there first)" >&2; exit 1; }
        cat > /etc/systemd/system/tf-cores.service <<'UNIT'
[Unit]
Description=Keep the engine cores (15-19) free of storage IRQs and at max clocks
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/tf-cores apply
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
        systemctl daemon-reload && systemctl enable --now tf-cores.service >/dev/null
        echo "installed: /usr/local/sbin/tf-cores, tf-cores.service ($(systemctl is-enabled tf-cores.service), $(systemctl is-active tf-cores.service))"
        ;;
    *) echo "usage: $0 [check|apply|install]" >&2; exit 2 ;;
esac
