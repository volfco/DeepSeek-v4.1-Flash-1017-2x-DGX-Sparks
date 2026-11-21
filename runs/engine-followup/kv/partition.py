"""Pair a low occupancy part-1 shared target with three 300K retained states on part 0."""
import argparse
import pathlib
import sys

here=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(here))
import pressure as p


def main():
    c = p.client
    c.args = argparse.Namespace(arm="kv-partition",salt="5",occupancy=True,replay_depth=False,finish_grace=False)
    c.health = p.health
    p.trial.out = here
    settings = p.trial.identity()
    p.trial.idle()
    if any(flags for _,flags in settings) or sum(p.rows())>32768 or min(c.memory())<5.2:
        raise RuntimeError("Expected reset production and 300K memory headroom")
    _,cases=p.history()
    end="Summarize the preceding context in detail."
    # The two recovered states shared/f2 have verified owner 0; f0 has owner 1.
    # Fill a new state while part 1 holds f0 so part 0 has more free space.
    source=cases["f0"].split("\n",1)[1]
    cases["f3"]="[review-partition-followup-3]\n"+source
    prompts=[cases["f0"].removesuffix(end)+task for task in c.tasks]
    c.emit("start",settings=settings,rows=p.rows(),target="f0",fillers=["f3","shared","f2"])
    try:
        p.probe("f0","target",3,cases["f0"],p.grid)
        before=p.rows()
        if not (before[0] <= 32768 and before[1] == p.grid):
            raise RuntimeError(f"Unexpected target placement {before}")
        def burst(label):
            reqs = c.cell(300000,"admission",prompts,list(range(4200,4204)),budget=1024,rep=3,
                          sharing=label,expected_cached=p.grid)
            return [(r["input_sha256"],r["tokens"],r["done"]["sha"]) for r in reqs]
        expected=burst("low-part1")
        for key in ("f3","shared","f2"):
            p.probe(key,"fill",3,cases[key])
            p.probe(key,"pre-pressure",3,cases[key],p.grid)
        peak=p.rows()
        if not (3*p.grid<=peak[0]<=3*p.grid+32768 and peak[1]==p.grid):
            raise RuntimeError(f"Could not construct 3:1 occupancy {peak}")
        c.emit("stage",stage="peak",rep=3,rows=peak)
        if burst("peak-lopsided")!=expected:
            raise RuntimeError("Burst token identity differs")
        for key in ("shared","f3","f2"):
            p.probe(key,"after-pressure",3,cases[key])
    finally:
        p.trial.idle()
        p.trial.restart("partition-restore",{})
        if p.trial.identity()!=settings:
            raise RuntimeError("Restored settings differ")
        p.trial.validate("partition-restore")
        p.trial.idle()
        c.emit("restored",settings=settings,health=p.health(),memory_gib=c.memory())
    c.emit("complete")


if __name__=="__main__":
    main()
