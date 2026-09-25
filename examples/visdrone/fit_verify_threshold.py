"""Fit the verifier's score threshold on the development panel, freeze it, then score held-out once.

The verifier's detection score already includes SAM 3's presence head (the processor multiplies the
per-query score by the presence probability), so raising the threshold above the 0.3 the stage ran
at is presence filtering, not a new mechanism. Scores below 0.3 were never recorded, so 0.3 is the
floor of the sweep.

Fitting is a sensitivity target, not a precision optimum: precision here is a bound, because an
unmatched box is either false or a real object the reference omits, and optimising a bound would
reward deleting unannotated objects. Sensitivity on known positives is the one quantity the
reference measures without that ambiguity, so each operating point is the highest threshold that
keeps sensitivity at or above its target on development data. Per-class thresholds use the
pipeline's own label, the only class available at inference, and fall back to the global threshold
where a class has too few known positives to fit.

The thresholds are written before the held-out panel is read (D57).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.evaluation import match  # noqa: E402
from build_benchmark import references, select_images  # noqa: E402

FLOOR = 0.3


def rows_for(runs, root, mapping, seed, count):
    """One row per retained object: pipeline label, SAM 3 score (None if unconfirmed), known status."""
    rows, eligible_total = [], 0
    for name in select_images(root, count, seed):
        stem = Path(name).stem
        result = json.loads((Path(runs) / stem / 'annotation.json').read_text())
        records = {}
        for line in (Path(runs) / stem / 'evidence/candidates.jsonl').read_text().splitlines():
            record = json.loads(line).get('verification')
            if record:
                records[record['object']] = record
        retained = [{'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id'], 'id': o['id']} for o in result['objects']]
        refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'],
                             *result['image']['normalized_size'])
        eligible = [r for r in refs if r.get('eligible', True)]
        eligible_total += len(eligible)
        known = {a for a, _, _ in match(retained, eligible, .5)}
        for position, box in enumerate(retained):
            record = records.get(box['id'], {})
            rows.append({'image': stem, 'id': box['id'], 'label': box['label'], 'known': position in known,
                         'score': record.get('verifier_score') if record.get('confirmed') else None})
    return rows, eligible_total


def keep(row, rule):
    threshold = rule['per_class'].get(row['label'], rule['global'])
    return row['score'] is not None and row['score'] >= threshold


def highest_threshold(rows, target):
    """Highest threshold whose sensitivity on known positives stays >= target, never below the floor."""
    scores = sorted(r['score'] for r in rows if r['known'] and r['score'] is not None)
    positives = sum(r['known'] for r in rows)
    if not positives:
        return None
    best = FLOOR
    for s in scores:
        if s < FLOOR:
            continue
        if sum(x >= s for x in scores) / positives >= target:
            best = max(best, s)
    return best


def summary(rows, rule, eligible):
    kept = [r for r in rows if keep(r, rule)]
    positives = sum(r['known'] for r in rows)
    hit = sum(r['known'] for r in kept)
    undecided = len(rows) - positives
    return {'kept': len(kept), 'sensitivity': round(hit / positives, 4),
            'undecided_confirmed': round((len(kept) - hit) / undecided, 4) if undecided else None,
            'recall': round(hit / eligible, 4), 'precision': round(hit / len(kept), 4) if kept else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict-v2.json'))
    parser.add_argument('--dev-runs', required=True)
    parser.add_argument('--dev-seed', type=int, default=0)
    parser.add_argument('--heldout-runs', required=True)
    parser.add_argument('--heldout-seed', type=int, default=1)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--targets', default='0.98,0.95,0.90,0.85')
    parser.add_argument('--min-class-positives', type=int, default=20)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    targets = [float(t) for t in args.targets.split(',')]

    dev, dev_eligible = rows_for(args.dev_runs, root, mapping, args.dev_seed, args.count)
    classes = sorted({r['label'] for r in dev})
    rules = {'current (0.3)': {'global': FLOOR, 'per_class': {}}}
    for target in targets:
        global_threshold = highest_threshold(dev, target)
        rules[f'global @ sens {target}'] = {'global': global_threshold, 'per_class': {}}
        per_class = {}
        for label in classes:
            subset = [r for r in dev if r['label'] == label]
            if sum(r['known'] for r in subset) >= args.min_class_positives:
                per_class[label] = highest_threshold(subset, target)
        rules[f'per-class @ sens {target}'] = {'global': global_threshold, 'per_class': per_class}

    frozen = {'fitted_on': {'runs': args.dev_runs, 'seed': args.dev_seed, 'count': args.count},
              'floor': FLOOR, 'min_class_positives': args.min_class_positives, 'rules': rules,
              'development': {name: summary(dev, rule, dev_eligible) for name, rule in rules.items()}}
    out = Path(args.out)
    out.write_text(json.dumps(frozen, indent=2))  # frozen before held-out is read

    heldout, heldout_eligible = rows_for(args.heldout_runs, root, mapping, args.heldout_seed, args.count)
    frozen['heldout'] = {name: summary(heldout, rule, heldout_eligible) for name, rule in rules.items()}
    frozen['heldout_scored_from'] = {'runs': args.heldout_runs, 'seed': args.heldout_seed}
    out.write_text(json.dumps(frozen, indent=2))

    print(f'development {len(dev)} boxes, {dev_eligible} eligible; held-out {len(heldout)} boxes, {heldout_eligible} eligible\n')
    print(f'{"rule":24s} | {"dev sens":>8s} {"recall":>7s} {"prec":>6s} | {"held sens":>9s} {"recall":>7s} {"prec":>6s} {"kept":>5s}')
    for name, rule in rules.items():
        d, h = frozen['development'][name], frozen['heldout'][name]
        print(f'{name:24s} | {d["sensitivity"]:8.4f} {d["recall"]:7.4f} {d["precision"]:6.4f} | '
              f'{h["sensitivity"]:9.4f} {h["recall"]:7.4f} {h["precision"]:6.4f} {h["kept"]:5d}')
    for name, rule in rules.items():
        if rule['per_class']:
            print(f'  {name}: global {rule["global"]:.3f}, ' +
                  ', '.join(f'{k} {v:.3f}' for k, v in sorted(rule['per_class'].items())))
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
