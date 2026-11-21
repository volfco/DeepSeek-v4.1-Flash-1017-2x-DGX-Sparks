"""Freeze two untouched excerpts from each of 20 additional tracked files before scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / 'tools'))
import score_human_text


def freeze(previous, out):
    prior = [json.loads(line)['provenance'] for line in previous.read_text().splitlines()]
    revisions = {row['commit'] for row in prior}
    assert len(revisions) == 1
    revision = revisions.pop()
    excluded = {row['source'] for row in prior}
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision], text=True).splitlines()
    for kind in ('prose', 'code'):
        eligible = [p for p in files if p not in excluded and
                    (p.endswith('.md') if kind == 'prose' else p.startswith(('tools/', 'scripts/')) and p.endswith('.py'))]
        ordered = sorted(eligible, key=lambda p: hashlib.sha256(('overnight-prune-confirmation/' + p).encode()).digest())
        sources = 0
        for name in ordered:
            data = subprocess.check_output(['git', 'show', revision+':'+name])
            splits = list(score_human_text.splits(data, 1280, 248, 4))
            if len(splits) < 2:
                continue
            for j,(p0,p1,p2) in enumerate(splits[1:3] if len(splits)>=3 else splits[:2]):
                row = dict(id=f'{kind}-extra-{sources:02d}-{j}',kind=kind,
                           prompt_text=data[p0:p1].decode(),continuation_text=data[p1:p2].decode(),
                           provenance=dict(source=name,commit=revision,sha256=hashlib.sha256(data).hexdigest(),
                                           prompt_bytes=[p0,p1],continuation_bytes=[p1,p2]))
                print(json.dumps(row),file=out)
            sources += 1
            if sources == 10:
                break
        if sources != 10:
            raise RuntimeError(f'Found only {sources} eligible disjoint {kind} files')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    ap.add_argument('--previous',type=Path,default=Path('runs/engine-followup/reference-text.jsonl'))
    args=ap.parse_args()
    temp=args.out.with_suffix('.writing')
    if args.out.exists():
        raise FileExistsError(args.out)
    with temp.open('x') as out:
        freeze(args.previous,out)
    temp.replace(args.out)
