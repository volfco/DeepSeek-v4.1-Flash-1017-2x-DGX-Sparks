#!/usr/bin/env python3
"""Put `requests running` on a right-hand axis of the Tensorfold dashboard's prompt / generation tok/s panels.

Each target panel becomes an OpenObserve custom chart (type "custom_chart", JS in customChartContent; the JS runs in
a sandboxed iframe, gets `data` = one {result:[{metric, values:[[sec,"v"]]}]} per PromQL query, and must assign
`option`). The panel keeps its queries, legends and benchmark lines on the left axis; sum(requests_running) is
appended as the last query and drawn as a step line on the right axis.

    . runs/observe-review-20261123/scripts/o2env.sh
    python3 -I runs/observe-20261126/scripts/dash_running.py build OUT.json    # new definition + per-panel JS files
    python3 -I runs/observe-20261126/scripts/dash_running.py test  OUT.json    # run every panel's JS in node on live data
    python3 -I runs/observe-review-20261123/scripts/dash_api.py put 7522360476127199234 OUT.json
Undo: dash_api.py restore runs/observe-20261126/data/tensorfold-dash-before.json
"""
import copy
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../observe-review-20261123/scripts"))
import dash_api  # noqa: E402
import o2  # noqa: E402

DASH = "7522360476127199234"
TARGETS = ["Panel_ID0000013", "Panel_ID0000014", "Panel_R0008", "Panel_R0009", "Panel_R0014", "Panel_R0029"]
RUNNING = "sum(tensorfold:requests_running)"
MARK = "// requests-running overlay (runs/observe-20261126/scripts/dash_running.py)"

JS = MARK + r"""
// 'data': one entry per query, in the panel's query order; the last query is requests running (right axis).
var Q = __SPEC__;
var series = [];
for (var i = 0; i < data.length; i++) {
  var d = data[i];
  var q = Q[i] || { text: "query " + (i + 1), axis: 0, ref: false };
  if (!d || !d.result) continue;
  for (var j = 0; j < d.result.length; j++) {
    var r = d.result[j];
    var name = q.lab ? (r.metric[q.lab] || q.text) : q.text;
    var pts = [];
    for (var k = 0; k < r.values.length; k++) pts.push([r.values[k][0] * 1000, Number(r.values[k][1])]);
    var s = { name: name, type: "line", data: pts, showSymbol: false, yAxisIndex: q.axis,
              lineStyle: { width: q.ref ? 1 : 1.5, type: q.ref ? "dashed" : "solid" } };
    if (q.axis === 1) { s.step = "end"; s.areaStyle = { opacity: 0.08 }; }
    series.push(s);
  }
}
option = {
  tooltip: { trigger: "axis" },
  legend: { type: "scroll", bottom: 0 },
  grid: { left: 64, right: 64, top: 32, bottom: 56 },
  xAxis: { type: "time" },
  yAxis: [
    { type: "value", name: __UNIT__, min: 0 },
    { type: "value", name: "requests running", min: 0, minInterval: 1, splitLine: { show: false } }
  ],
  series: series
};
"""


def legend_spec(q):
    """The panel's promql_legend as the JS takes it: a label to read ({ref}, {host_name}) or fixed text."""
    leg = (q.get("config") or {}).get("promql_legend") or ""
    is_ref = "label_replace(vector(" in q["query"]
    if leg.startswith("{") and leg.endswith("}") and leg.count("{") == 1:
        return {"lab": leg[1:-1], "text": q["query"][:60], "axis": 0, "ref": is_ref}
    return {"text": leg or q["query"][:60], "axis": 0, "ref": is_ref}


def running_query(template):
    q = copy.deepcopy(template)
    q["query"] = RUNNING
    q["fields"]["stream"] = RUNNING
    q["config"]["promql_legend"] = "requests running"
    return q


def convert(p):
    if p["queries"][-1]["query"] == RUNNING:  # already converted: rebuild from its own queries
        p["queries"] = p["queries"][:-1]
    spec = [legend_spec(q) for q in p["queries"]]
    # the Overview panels' only legend is {host_name}; name the series for what they are
    if p["title"] in ("Prompt tokens/s", "Generation tokens/s"):
        spec[0] = {"text": p["title"].replace("/s", " a second, all requests"), "axis": 0, "ref": False}
    spec.append({"text": "requests running", "axis": 1, "ref": False})
    p["queries"].append(running_query(p["queries"][0]))
    cfg = p["config"]
    unit = cfg.get("unit_custom") if cfg.get("unit") == "custom" else "tok/s"
    p["type"] = "custom_chart"
    p["customChartContent"] = JS.replace("__SPEC__", json.dumps(spec)).replace("__UNIT__", json.dumps(unit or "tok/s"))
    note = "Right axis: requests running (sum(tensorfold:requests_running), 30 s gauge; the engine batches up to 8)."
    if note not in (p.get("description") or ""):
        p["description"] = ((p.get("description") or "") + " " + note).strip()
    return p


def build(out):
    reply = dash_api.get(DASH)
    d = copy.deepcopy(dash_api.bare(reply))
    done = []
    for t in d["tabs"]:
        for p in t["panels"]:
            if p["id"] in TARGETS:
                convert(p)
                done.append((t["name"], p["id"], p["title"]))
    missing = set(TARGETS) - {x[1] for x in done}
    if missing:
        raise SystemExit(f"panels not found: {sorted(missing)}")
    json.dump(d, open(out, "w"), indent=1)
    for x in done:
        print("converted", *x)


def test(out):
    """Every converted panel's JS: OpenObserve's own acorn checks are mirrored by a syntax parse, then the code runs in
    node with the panel's queries answered live over the last 6 h, and must yield an option with >= 2 series and the
    running series on axis 1."""
    d = json.load(open(out))
    end = time.time()
    start = end - 6 * 3600
    rc = 0
    for t in d["tabs"]:
        for p in t["panels"]:
            if p["id"] not in TARGETS:
                continue
            data = [o2.prom_range(q["query"], start, end, step=60)["data"] for q in p["queries"]]
            harness = ("const DATA = JSON.parse(require('fs').readFileSync(0,'utf8')); let option;\n"
                       + "(function(data){\n" + p["customChartContent"] + "\n})(DATA);"
                       + "\nconst s=option.series; const run=s.filter(x=>x.yAxisIndex===1);"
                       + "console.log(JSON.stringify({n:s.length, names:s.map(x=>x.name), run:run.length,"
                       + " runmax: run.length? Math.max(...run[0].data.map(v=>v[1])):null, axes: option.yAxis.map(a=>a.name),"
                       + " pts: s.map(x=>x.data.length)}));")
            r = subprocess.run(["node", "-e", harness], input=json.dumps(data), capture_output=True, text=True)
            ok = r.returncode == 0 and '"run":1' in r.stdout
            rc |= not ok
            print("OK " if ok else "BAD", p["id"], p["title"], r.stdout.strip() or r.stderr.strip()[:400])
    sys.exit(rc)


if __name__ == "__main__":
    {"build": build, "test": test}[sys.argv[1]](sys.argv[2])
