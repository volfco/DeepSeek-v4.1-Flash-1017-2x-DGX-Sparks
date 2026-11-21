"""Close block s01 after the runner was stopped inside it: add what the runner adds at a block's end (samples, the
engine's done lines, status) and the narrow probe that was run by hand on the same process. Run on the head, from the
recipe root, before the pair is restarted."""
import importlib.util, json, os, time
os.environ.update(CAMPAIGN_OUT="runs/suite-20261119", CAMPAIGN_IMAGE="exl3-tp2-ea522d353626", CAMPAIGN_PROD="{}", CAMPAIGN_CLOCK_ABORT="0")
spec = importlib.util.spec_from_file_location("run_suite", "runs/suite-20261119/run_suite.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
bo = m.bo
prefix = f"{bo.BLK}/s01"; path = prefix + ".json"
doc = json.load(open(path))
if "status" in doc:
    raise SystemExit("s01 already has a status")
doc["samples"] = {"head": bo.Sampler.summ(prefix + ".head.samples"), "worker": bo.Sampler.summ(prefix + ".worker.samples")}
doc["banners"], doc["done_lines"] = bo.engine_logs(prefix)
doc["t_end"] = time.time()
s = doc["samples"]
doc["clock_valid"] = max(s["head"].get("clock_max", 0), s["worker"].get("clock_max", 0)) <= bo.CLOCK_MAX_OK and s["head"].get("n", 0) > 0 and s["worker"].get("n", 0) > 0
rows = json.load(open(prefix + ".probe.json")) if os.path.exists(prefix + ".probe.json") else []
doc["cells"]["probe"] = {"rc": 0 if rows and not any(r.get("failed") for r in rows) else 2, "narrow": True,
                         "rows": [{k: v for k, v in r.items() if k != "per_round"} for r in rows]}
doc["cut_short"] = "the runner was stopped inside this block to shorten the suite; the narrow probe was then run by hand on the same process"
doc["status"] = "ok"
bo.save(path, doc)
print("s01 closed: cells", list(doc["cells"]), "done lines", len(doc["done_lines"]), "probe rows", len(rows), "clock_valid", doc["clock_valid"])
