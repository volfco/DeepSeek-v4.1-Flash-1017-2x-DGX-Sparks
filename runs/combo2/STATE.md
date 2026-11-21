# Head state during the 0129-0145 campaign (2026-11-16)

- The head's zig/patches/ holds 0001-0145, so ./start.sh builds and runs image exl3-tp2-bd9020f403bd (the new code).
- The tag exl3-tp2-b40ea899e367 currently also holds the NEW code. A start with IMAGE=...b40ea899e367 rebuilt it
  from the 0001-0145 patch set at 16:19Z, and the old build was deleted.
- To return production to the 0001-0128 image:
  1. Move zig/patches/01{29..45}-*.patch out of zig/patches/ on the head.
  2. Run `docker rmi tensorfold-dsv41:exl3-tp2-b40ea899e367` on both Sparks.
  3. Run `PATH=~/.local/bin:$PATH ./scripts/prepare.sh` (about 6 min of Zig builds).
  4. Run ./start.sh restart, then check that TF_DS_RECIPE_PATCHES=b40ea899e367 is in the container env.
