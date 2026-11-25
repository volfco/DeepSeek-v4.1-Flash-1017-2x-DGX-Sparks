#!/usr/bin/env python3
"""Dashboard read/write helpers and the restore command. Credentials from the environment (. scripts/o2env.sh).

    python3 -I scripts/dash_api.py export <dashboard_id> <out.json>          # GET, the API's reply as is
    python3 -I scripts/dash_api.py restore <backup.json>                     # PUT the saved definition back (undo)
    python3 -I scripts/dash_api.py put <dashboard_id> <definition.json>      # PUT a bare definition with the live hash
    python3 -I scripts/dash_api.py diff <a.json> <b.json>                    # compare two definitions, server fields aside

`restore` reads the saved GET reply (dashboards-before/<id>.json), takes its v<version> definition and PUTs it to the
same id with the dashboard's current hash, so it works however many times the dashboard was changed since.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o2  # noqa: E402

FOLDER = "default"
SERVER_FIELDS = ("dashboardId", "created", "owner", "role", "updatedAt")


def get(dash_id, folder=FOLDER):
    return o2.http("GET", f"/api/{o2.ORG}/dashboards/{dash_id}", params={"folder": folder})


def bare(reply):
    """The definition inside a GET reply ({"v8": {...}, "version": 8, "hash": ...}); a bare definition passes through."""
    if "version" in reply and f"v{reply['version']}" in reply and isinstance(reply[f"v{reply['version']}"], dict):
        return reply[f"v{reply['version']}"]
    return reply


def create(definition, folder=FOLDER):
    return o2.http("POST", f"/api/{o2.ORG}/dashboards", definition, {"folder": folder})


def put(dash_id, definition, folder=FOLDER):
    live = get(dash_id, folder)
    return o2.http("PUT", f"/api/{o2.ORG}/dashboards/{dash_id}", definition, {"folder": folder, "hash": live["hash"]})


def delete(dash_id, folder=FOLDER):
    return o2.http("DELETE", f"/api/{o2.ORG}/dashboards/{dash_id}", params={"folder": folder})


def norm(d):
    d = json.loads(json.dumps(bare(d)))
    for k in SERVER_FIELDS:
        d.pop(k, None)
    return d


def diff(a, b, path=""):
    """Differences between two JSON values as (path, a, b); numbers equal as floats are equal (1 == 1.0)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                if b[k] not in (None, "", [], {}, False):
                    out.append((f"{path}.{k}", "<absent>", b[k]))
            elif k not in b:
                if a[k] not in (None, "", [], {}, False):
                    out.append((f"{path}.{k}", a[k], "<absent>"))
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path + "[len]", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    else:
        same = a == b or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool)
                          and not isinstance(b, bool) and float(a) == float(b))
        if not same:
            out.append((path, a, b))
    return out


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "export":
        json.dump(get(a[1]), open(a[2], "w"), indent=1)
        print("saved", a[2])
    elif a[0] == "restore":
        saved = json.load(open(a[1]))
        d = bare(saved)
        r = put(d["dashboardId"], d)
        back = get(d["dashboardId"])
        dd = diff(norm(saved), norm(back))
        print("restored", d["dashboardId"], repr(d.get("title")), "hash", back.get("hash"), "differences from the backup:", len(dd))
        for x in dd[:20]:
            print("  ", x)
    elif a[0] == "put":
        d = bare(json.load(open(a[2])))
        put(a[1], d)
        print("updated", a[1], "hash", get(a[1]).get("hash"))
    elif a[0] == "diff":
        dd = diff(norm(json.load(open(a[1]))), norm(json.load(open(a[2]))))
        print("differences:", len(dd))
        for x in dd[:40]:
            print("  ", str(x)[:300])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
