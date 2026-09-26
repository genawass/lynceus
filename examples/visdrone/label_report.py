"""How well a panel is named, not just found: exact class, parent fallback, or wrong, per reference class.

Finding a box and naming it are separate questions once an ontology has more than a handful of
classes. For every retained box matching an eligible reference (IoU >= 0.5), the pipeline's label is
scored against the reference's exact class, the broader labels the mapping permits, or neither; the
verifier's own label is scored the same way where it confirmed the box. Rows can be restricted to
boxes the verifier kept at a threshold, so naming is reported for the set a consumer would receive.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.evaluation import match  # noqa: E402
from build_benchmark import references, select_images  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--threshold', type=float, help='keep only boxes the verifier confirmed at or above this score')
    parser.add_argument('--out')
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    exact_of = {c['name']: c['permitted_labels'][0] for c in mapping['categories'].values() if c.get('eligible', True)}
    confusion = defaultdict(Counter)
    verdicts = Counter()
    verifier = Counter()
    per_class = defaultdict(Counter)
    for name in select_images(root, args.count, args.seed):
        stem = Path(name).stem
        result = json.loads((Path(args.runs) / stem / 'annotation.json').read_text())
        records = {}
        for line in (Path(args.runs) / stem / 'evidence/candidates.jsonl').read_text().splitlines():
            rec = json.loads(line).get('verification')
            if rec:
                records[rec['object']] = rec
        refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'],
                             *result['image']['normalized_size'])
        eligible = [r for r in refs if r.get('eligible', True)]
        objects = [o for o in result['objects']]
        for a, b, _ in match([{'bbox_xyxy': o['bbox_xyxy']} for o in objects], eligible, .5):
            obj, ref = objects[a], eligible[b]
            rec = records.get(obj['id'], {})
            score = rec.get('verifier_score') if rec.get('confirmed') else None
            if args.threshold is not None and (score is None or score < args.threshold):
                continue
            label, reference = obj['label']['id'], ref['visdrone_category']
            exact = exact_of[reference]
            verdict = 'exact' if label == exact else 'broader' if label in ref['permitted_labels'] else 'wrong'
            verdicts[verdict] += 1
            per_class[reference][verdict] += 1
            confusion[reference][label] += 1
            if score is not None:
                vlabel = rec.get('verifier_label')
                verifier['exact' if vlabel == exact else 'broader' if vlabel in ref['permitted_labels'] else 'wrong'] += 1

    total = sum(verdicts.values())
    labels = sorted({l for row in confusion.values() for l in row})
    report = {'mapping': mapping['id'], 'runs': args.runs, 'threshold': args.threshold, 'matched': total,
              'pipeline': {k: verdicts[k] for k in ('exact', 'broader', 'wrong')},
              'verifier': {k: verifier[k] for k in ('exact', 'broader', 'wrong')},
              'per_class': {c: dict(v) for c, v in per_class.items()},
              'confusion': {c: dict(v) for c, v in confusion.items()}}
    print(f"{mapping['id']}  matched boxes {total}" + (f"  (verifier score >= {args.threshold})" if args.threshold else ''))
    for who, counts in (('pipeline label', verdicts), ('SAM 3 label', verifier)):
        n = sum(counts.values()) or 1
        print(f"  {who:15s} exact {counts['exact'] / n:.3f}  broader {counts['broader'] / n:.3f}  wrong {counts['wrong'] / n:.3f}  (n={sum(counts.values())})")
    print('\n  reference \\ label  ' + '  '.join(f'{l[:12]:>12s}' for l in labels))
    for c in sorted(confusion, key=lambda c: -sum(confusion[c].values())):
        print(f'  {c[:18]:18s}  ' + '  '.join(f'{confusion[c][l]:12d}' for l in labels))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
