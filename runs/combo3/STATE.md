# Head state for combo3 (0160-0171), staged 2026-11-16

- `runs/combo3/chain.sh` runs detached on the head. When the combo2 runner (pid 507660) exits with CAMPAIGN DONE, it:
  1. copies `runs/combo3/newpatches/` (0160-0171 and README) into `zig/patches/`
  2. checks the full set against `patches.sha256`
  3. runs `prepare.sh` and builds image `exl3-tp2-9aa632485fe4`
  4. launches the runner with `CAMPAIGN_OUT=runs/combo3`
  Its progress goes to `runs/combo3/chain.log`.
- To return production to the 0001-0128 image afterwards, move `zig/patches/01{29..71}-*.patch` out of the head's `zig/patches/`, then follow `runs/combo2/STATE.md` steps 2-4.
