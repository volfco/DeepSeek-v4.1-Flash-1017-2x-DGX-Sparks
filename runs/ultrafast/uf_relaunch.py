#!/usr/bin/env python3
"""Print a `docker run` command line (shell-quoted) that recreates a container from its saved `docker inspect` JSON, with a
new name, extra mounts and a replaced command.
usage: uf_relaunch.py INSPECT.json NAME [--mount SRC:DST[:MODE]]... -- CMD...   (CMD empty: keep the container's)"""
import json, shlex, sys

a = sys.argv[1:]
c = json.load(open(a[0]))[0]
name = a[1]
rest = a[2:]
mounts, cmd = [], None
while rest:
    if rest[0] == "--mount":
        mounts.append(rest[1]); rest = rest[2:]
    elif rest[0] == "--":
        cmd = rest[1:]; rest = []
    else:
        raise SystemExit("bad argument " + rest[0])
h = c["HostConfig"]
argv = ["docker", "run", "-d", "--name", name, "--gpus", "all"]
if h.get("IpcMode"): argv += ["--ipc", h["IpcMode"]]
if h.get("NetworkMode"): argv += ["--network", h["NetworkMode"]]
for u in h.get("Ulimits") or []:
    argv += ["--ulimit", f"{u['Name']}={u['Soft']}:{u['Hard']}"]
for cap in h.get("CapAdd") or []:
    argv += ["--cap-add", cap.replace("CAP_", "")]
for d in h.get("Devices") or []:
    argv += ["--device", f"{d['PathOnHost']}:{d['PathInContainer']}"]
if h.get("CpusetCpus"): argv += ["--cpuset-cpus", h["CpusetCpus"]]
for m in c["Mounts"]:
    src = m["Name"] if m["Type"] == "volume" else m["Source"]
    argv += ["-v", f"{src}:{m['Destination']}" + ("" if m.get("RW", True) else ":ro")]
for m in mounts:
    argv += ["-v", m]
for e in c["Config"]["Env"]:
    argv += ["-e", e]
argv.append(c["Config"]["Image"])
argv += cmd if cmd else (c["Config"]["Cmd"] or [])
print(shlex.join(argv))
