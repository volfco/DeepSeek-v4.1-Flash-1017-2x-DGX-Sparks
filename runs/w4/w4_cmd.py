#!/usr/bin/env python3
"""Print a `docker run` line (shell-quoted) for one rank of a W4 batch-driver cell (`tf-dsv41-lanes` as both ranks),
recreated from the served containers' saved `docker inspect` JSON - runs/ultrafast/uf_prof.sh's recipe, with the spill
tier stripped (TF_DS_SPILL_*, TF_SPILL_* and the /spill mount) so a batch run never touches the served store.

usage: w4_cmd.py RANK INSPECT0 INSPECT1 NAME --graphs G --profile P [--requests FILE --parallel N --reqdir DIR]
                 [--env K=V]... [--unset K]...
RANK 1: rank 1's own container and command (graphs replaced, --profile appended).
RANK 0: rank 0's container (mounts, env, cpuset) with rank 1's lanes command made rank 0's (uf_prof.sh's mapping)."""
import json, re, shlex, sys

a = sys.argv[1:]
rank, i0p, i1p, name = int(a[0]), a[1], a[2], a[3]
rest = a[4:]
opt = {"--graphs": None, "--profile": None, "--requests": None, "--parallel": None, "--reqdir": None}
env_add, unset = [], set()
while rest:
    k = rest[0]
    if k == "--env":
        env_add.append(rest[1])
    elif k == "--unset":
        unset.add(rest[1])
    elif k in opt:
        opt[k] = rest[1]
    else:
        raise SystemExit("bad argument " + k)
    rest = rest[2:]

i0 = json.load(open(i0p))[0]
i1 = json.load(open(i1p))[0]
c = i0 if rank == 0 else i1
env0 = dict(e.split("=", 1) for e in i0["Config"]["Env"])
cmd = list(i1["Config"]["Cmd"])
if cmd[0] != "tf-dsv41-lanes":
    raise SystemExit("rank 1's command is not tf-dsv41-lanes: " + cmd[0])
if rank == 0:
    cmd = [cmd[0], cmd[1], env0["TF_DS_RANK_CACHE"], "0", cmd[4], cmd[5], cmd[6], cmd[7]] + cmd[8:]
    cmd[cmd.index("--rdma") + 1] = env0["TF_RDMA_DEVICES"]
cmd[cmd.index("--graphs") + 1] = opt["--graphs"]
if rank == 0 and opt["--requests"]:
    cmd += ["--requests", "/reqs/" + opt["--requests"], "--parallel", opt["--parallel"]]
cmd += ["--profile", opt["--profile"]]

drop = re.compile(r"^(TF_DS_SPILL_|TF_SPILL_)")
env = [e for e in c["Config"]["Env"] if not drop.match(e) and e.split("=", 1)[0] not in unset]
keys = {e.split("=", 1)[0] for e in env_add}
env = [e for e in env if e.split("=", 1)[0] not in keys] + env_add

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
    if m["Destination"] == "/spill":
        continue
    src = m["Name"] if m["Type"] == "volume" else m["Source"]
    argv += ["-v", f"{src}:{m['Destination']}" + ("" if m.get("RW", True) else ":ro")]
if rank == 0 and opt["--reqdir"]:
    argv += ["-v", f"{opt['--reqdir']}:/reqs:ro"]
for e in env:
    argv += ["-e", e]
argv.append(c["Config"]["Image"])
argv += cmd
print(shlex.join(argv))
