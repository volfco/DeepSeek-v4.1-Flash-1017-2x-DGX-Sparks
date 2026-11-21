import json, re, sys
d = json.load(sys.stdin)[0]
c = d["Config"]
red = lambda s: re.sub(r"(?i)^(.*(key|token|secret|pass).*?=).*$", r"\1<redacted>", s)
print("image=" + c["Image"])
print("started=" + d["State"]["StartedAt"])
args = c.get("Cmd") or []
out = []; hide = False
for a in args:
    out.append("<redacted>" if hide else a)
    hide = bool(re.search(r"(?i)key|token|secret|pass", a)) and a.startswith("--")
print("cmd=" + json.dumps(out))
for e in sorted(c.get("Env") or []):
    if re.match(r"^(TF_|TENSORFOLD_)", e) and not re.search(r"(?i)key|token|secret|pass", e.split("=")[0]):
        print(e)
