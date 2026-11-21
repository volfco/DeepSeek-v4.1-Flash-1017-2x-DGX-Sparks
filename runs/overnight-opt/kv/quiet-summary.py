"""Summarize only the covered production intervals from the completed aggregate observer."""
import argparse
import datetime
import json
import math
from pathlib import Path


root = Path(__file__).resolve().parent


def epoch(value):
    return datetime.datetime.fromisoformat(value).timestamp() if isinstance(value, str) else float(value)


def summarize(rows):
    previous_end = -math.inf
    for row in sorted(rows, key=lambda row: epoch(row['start'])):
        start, end = epoch(row['start']), epoch(row['end'])
        if not previous_end <= start < end or not isinstance(row['completed'], int) or row['completed'] < 0:
            raise ValueError('Overlapping, negative or invalid coverage')
        if 'seconds' in row and abs(float(row['seconds']) - (end - start)) > 0.001:
            raise ValueError('Recorded interval duration differs from wall timestamps')
        previous_end = end
    return {
        'intervals': len(rows),
        'seconds_covered': sum(epoch(row['end']) - epoch(row['start']) for row in rows),
        'completed': sum(row['completed'] for row in rows),
        'nonempty_intervals': sum(row['completed'] > 0 for row in rows),
        'longest_seconds': max((epoch(row['end']) - epoch(row['start']) for row in rows), default=0),
    }


def main(path):
    history = root.parent.parent / 'engine-followup/kv'
    prior = [json.loads((history/'archive-window.json').read_text()),
             json.loads((history/'summary.json').read_text())['traffic']]
    with (root/'pilot01.log').open() as f:
        initial = json.loads(next(f))
    if initial['kind'] != 'quiet':
        raise ValueError('Missing the separate pre-pilot interval')
    windows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {
        'earlier_two_windows': summarize(prior),
        'overnight_initial': summarize([initial]),
        'overnight_lock_free_sampled': {**summarize(windows),
            'sampled_running_max': max((r['running_max'] for r in windows), default=None),
            'sampled_waiting_max': max((r['waiting_max'] for r in windows), default=None)},
        'all_measured_disjoint_nov18': summarize([*prior, initial, *windows]),
    }
    if windows:
        result['overnight_lock_free_sampled']['first_sample_utc'] = datetime.datetime.fromtimestamp(
            min(r['start'] for r in windows), datetime.timezone.utc).isoformat()
        result['overnight_lock_free_sampled']['last_sample_utc'] = datetime.datetime.fromtimestamp(
            max(r['end'] for r in windows), datetime.timezone.utc).isoformat()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('intervals', type=Path)
    print(json.dumps(main(parser.parse_args().intervals), indent=2))
