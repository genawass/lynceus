"""Attribute every missed reference to the mechanism that lost it, and curve the score as a pruner.

Both diagnostics stay interpretable on VisDrone, which annotates only road users. Misses are fully
determined by the reference, so partitioning them assumes nothing about unannotated objects. And a
pruner's removals among matched boxes are exactly measurable even when its removals among
unmatched boxes are not, which is enough to tell a discriminating scorer from a random one.

The shipped run decides what counts as a miss. The probe run is the same schedule at a lower
threshold, and supplies the proposals the shipped threshold removed inside the adapter.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.diagnostics import (MISS_BUCKETS, attribute_misses, boundary_headroom,  # noqa: E402
                                 selection_separability, selectivity_curve)
from build_benchmark import references, select_images  # noqa: E402


def retained_candidates(run_dir):
    """The retained set with its detector scores.

    Read from the candidate store rather than from `objects`, because the score that ranked a box
    is evidence and does not appear in the annotation: the contract keeps model scores out of the
    result and in the evidence, so that nothing downstream can mistake one for a probability.
    """
    records = [json.loads(line) for line in (run_dir / 'evidence/candidates.jsonl').read_text().splitlines() if line]
    return [r for r in records if r['disposition'] == 'retained'], records


def table(title, rows, columns):
    width = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns} if rows else {}
    print(f'\n{title}')
    print('  '.join(c.ljust(width[c]) for c in columns))
    print('  '.join('-' * width[c] for c in columns))
    for row in rows:
        print('  '.join(str(row[c]).ljust(width[c]) for c in columns))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict.json'))
    parser.add_argument('--shipped', required=True, help='run directory tree at the shipped threshold')
    parser.add_argument('--probe', required=True, help='run directory tree at a lower threshold')
    parser.add_argument('--shipped-threshold', type=float, default=0.1)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--near-iou', type=float, default=0.3)
    parser.add_argument('--scores', help='JSON map of "image/candidate-id" to an external pruner score; '
                                         'without it the curve uses the detector score as the baseline')
    parser.add_argument('--out')
    args = parser.parse_args()

    external = json.loads(Path(args.scores).read_text()) if args.scores else None

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    stems = [Path(name).stem for name in select_images(root, args.count, args.seed)]

    totals = {bucket: 0 for bucket in MISS_BUCKETS}
    headroom_totals, headroom_by_size, headroom_rows = {}, {}, []
    separability = {'cases': 0, 'contention': 0, 'counts': {}}
    by_size, per_image, all_misses = {}, [], []
    panel_boxes, panel_scores, panel_refs, offset = [], [], [], 0
    panel_baseline, unscorable = [], 0
    for stem in stems:
        shipped_dir, probe_dir = Path(args.shipped) / stem, Path(args.probe) / stem
        with Image.open(root / 'images/val' / f'{stem}.jpg') as image:
            width, height = image.size
        refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'], width, height)
        eligible = [r for r in refs if r.get('eligible', True)]
        retained, shipped_records = retained_candidates(shipped_dir)
        _, probe = retained_candidates(probe_dir)

        report = attribute_misses(eligible, retained, probe, args.shipped_threshold,
                                  match_iou=args.match_iou, near_iou=args.near_iou)
        for bucket, count in report['counts'].items():
            totals[bucket] += count
        for size, counts in report['by_size'].items():
            row = by_size.setdefault(size, {b: 0 for b in MISS_BUCKETS})
            for bucket, count in counts.items():
                row[bucket] += count
        for miss in report['misses']:
            all_misses.append({**miss, 'image': stem})

        # What reconciliation could return at best, before any reconciliation rule is written.
        headroom = boundary_headroom(eligible, retained, probe, match_iou=args.match_iou, near_iou=args.near_iou)
        for verdict, count in headroom['counts'].items():
            headroom_totals[verdict] = headroom_totals.get(verdict, 0) + count
        for size, verdicts in headroom['by_size'].items():
            row = headroom_by_size.setdefault(size, {})
            for verdict, count in verdicts.items():
                row[verdict] = row.get(verdict, 0) + count
        headroom_rows.extend({**r, 'image': stem} for r in headroom['rows'])

        # Whether any signal the pipeline actually has would have reached that ceiling.
        separable = selection_separability(eligible, shipped_records, match_iou=args.match_iou)
        separability['cases'] += separable['cases']
        separability['contention'] += separable['contention']
        for signal, count in separable['counts'].items():
            separability['counts'][signal] = separability['counts'].get(signal, 0) + count
        per_image.append({'image': stem, 'eligible': report['eligible'], 'matched': report['matched'],
                          'missed': report['missed'], 'recall': report['recall'],
                          'retained': len(retained), 'probe_candidates': len(probe)})
        # The panel curve treats the ten images as one pool; matching stays per image because a
        # box can only match a reference in its own image, so references are offset into disjoint
        # coordinate space rather than pooled naively.
        # With an external scorer, boxes it cannot score leave both curves, so the pruner and the
        # baseline it must beat are compared on exactly the same boxes.
        curved = [r for r in retained if external is None or external.get(f'{stem}/{r["id"]}') is not None]
        unscorable += len(retained) - len(curved)
        panel_boxes.extend({**r, 'bbox_xyxy': [r['bbox_xyxy'][0] + offset, r['bbox_xyxy'][1],
                                               r['bbox_xyxy'][2] + offset, r['bbox_xyxy'][3]]} for r in curved)
        panel_scores.extend(external[f'{stem}/{r["id"]}'] if external else r['model_score'] for r in curved)
        panel_baseline.extend(r['model_score'] for r in curved)
        panel_refs.extend({**r, 'bbox_xyxy': [r['bbox_xyxy'][0] + offset, r['bbox_xyxy'][1],
                                              r['bbox_xyxy'][2] + offset, r['bbox_xyxy'][3]]} for r in eligible)
        offset += width + 10_000

    curve = selectivity_curve(panel_boxes, panel_refs, panel_scores, match_iou=args.match_iou)
    baseline = (selectivity_curve(panel_boxes, panel_refs, panel_baseline, match_iou=args.match_iou)
                if external else None)

    table('Per image', per_image, ['image', 'eligible', 'matched', 'missed', 'recall', 'retained', 'probe_candidates'])
    missed = sum(totals.values())
    table('Miss attribution (panel)',
          [{'bucket': b, 'misses': totals[b], 'share': f'{totals[b] / missed:.3f}' if missed else '-'}
           for b in MISS_BUCKETS],
          ['bucket', 'misses', 'share'])
    table('Miss attribution by reference size',
          [{'size_bin': size, **counts, 'total': sum(counts.values())} for size, counts in sorted(by_size.items())],
          ['size_bin', *MISS_BUCKETS, 'total'])
    verdicts = ['recoverable_by_selection', 'needs_new_geometry', 'not_localized']
    table('Boundary headroom: best IoU any recorded observation reached for a missed reference',
          [{'verdict': v, 'misses': headroom_totals.get(v, 0),
            'share': f'{headroom_totals.get(v, 0) / missed:.3f}' if missed else '-'} for v in verdicts],
          ['verdict', 'misses', 'share'])
    table('Boundary headroom by reference size',
          [{'size_bin': size, **{v: counts.get(v, 0) for v in verdicts}, 'total': sum(counts.values())}
           for size, counts in sorted(headroom_by_size.items())],
          ['size_bin', *verdicts, 'total'])

    table('Could any available signal have reached that ceiling?',
          [{'signal': s, 'cases_separated': separability['counts'].get(s, 0),
            'of_cases': separability['cases']}
           for s in ('higher_score', 'more_magnified', 'untruncated_over_truncated', 'no_signal_separates')],
          ['signal', 'cases_separated', 'of_cases'])
    print(f"  plus {separability['contention']} assignment-contention cases, which no representative rule addresses")

    table(('External pruner' if external else 'Detector score as a pruner (baseline any external pruner must beat)'),
          curve['rows'], ['threshold', 'kept', 'removed', 'matched_kept', 'matched_removed_rate',
                          'unmatched_removed_rate', 'selectivity', 'recall', 'precision'])
    print(f"\nboxes {curve['boxes']}, matched {curve['matched']}, unmatched {curve['unmatched']}")
    print(f"best selectivity: {curve['best']}")
    if baseline:
        print(f"\n{unscorable} retained boxes carried no class claim and left both curves")
        print(f"detector-score baseline on the SAME boxes: {baseline['best']}")
        gap = curve['best']['selectivity'] - baseline['best']['selectivity']
        print(f"external pruner beats the detector score by {gap:+.4f} selectivity")
    print(f"\nnever_proposed is the only bucket another detection route can fix: "
          f"{totals['never_proposed']} of {missed} misses "
          f"({totals['never_proposed'] / missed:.1%})" if missed else "\nno misses")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'panel': {'seed': args.seed, 'count': args.count, 'images': stems},
             'shipped_threshold': args.shipped_threshold, 'match_iou': args.match_iou,
             'near_iou': args.near_iou, 'per_image': per_image, 'counts': totals,
             'by_size': by_size, 'misses': all_misses, 'selectivity': curve,
             'headroom': {'counts': headroom_totals, 'by_size': headroom_by_size, 'rows': headroom_rows},
             'separability': separability, 'external_scores': bool(external),
             'baseline_selectivity': baseline, 'unscorable_boxes': unscorable},
            indent=1))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
