"""Compare frozen-policy arena traces, ignoring timing/Q metadata only."""
import argparse
import gzip
import json

from pathlib import Path


def events(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [{key: value for key, value in json.loads(line).items() if key != 'meta'}
                for line in stream]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    args = parser.parse_args()
    left = {path.name: path for path in (args.left / 'games').glob('*.json.gz')}
    right = {path.name: path for path in (args.right / 'games').glob('*.json.gz')}
    assert left and left.keys() == right.keys(), 'game set mismatch'
    mismatches = []
    for name in sorted(left):
        a, b = events(left[name]), events(right[name])
        if a != b:
            first = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
            mismatches.append(dict(file=name, event_index=first,
                                   left=a[first:first+1], right=b[first:first+1]))
    result = dict(games=len(left), identical=len(left)-len(mismatches), mismatches=mismatches)
    (args.left / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    assert not mismatches, 'trajectory differences require investigation before engine rollout'


if __name__ == '__main__':
    main()
