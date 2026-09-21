"""What did refining boxes against pixels actually return?

Refinement adds no box and removes none, so every change in recall comes from geometry. That makes
the comparison narrow and fair: the same retained set, the same matching rule, the same panel,
with only the coordinates replaced.

The headline number is not mean IoU but how many missed references cross the match criterion,
because that is what the recall gate counts and what miss attribution said was at stake.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.diagnostics import size_bin  # noqa: E402
from lynceus.evaluation import match  # noqa: E402
from lynceus.geometry import iou  # noqa: E402
from build_benchmark import references, select_images  # noqa: E402


def pick(refined, original, min_quality):
    """The refined box, or the original where refinement is absent or too weakly supported.

    A refined entry may carry the model's own mask quality. Gating on it is a rule the pipeline
    could actually run, because that number is available at inference; the threshold itself is
    still a choice that has to be frozen on development data before it means anything.
    """
    if refined is None:
        return original
    if isinstance(refined, dict):
        if min_quality is not None and refined.get('quality', 0.0) < min_quality:
            return original
        return refined['box']
    return refined


def summarise(boxes, eligible, match_iou):
    pairs = match(boxes, eligible, match_iou)
    return {'matched': len(pairs), 'boxes': len(boxes), 'hit': {b for _, b, _ in pairs},
            'mean_iou': (sum(v for _, _, v in pairs) / len(pairs)) if pairs else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict.json'))
    parser.add_argument('--runs', required=True)
    parser.add_argument('--refined', required=True, help='JSON map of "<stem>/<candidate id>" to a refined box')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--min-quality', type=float, default=None,
                        help="apply a refinement only when the model's own mask quality reaches this")
    parser.add_argument('--out')
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    stems = [Path(name).stem for name in select_images(root, args.count, args.seed)]
    refined_boxes = json.loads(Path(args.refined).read_text())

    totals = {'eligible': 0, 'before': 0, 'after': 0, 'recovered': 0, 'lost': 0, 'boxes': 0}
    iou_before, iou_after, by_size, rows = [], [], {}, []
    for stem in stems:
        records = [json.loads(line) for line
                   in (Path(args.runs) / stem / 'evidence/candidates.jsonl').read_text().splitlines() if line]
        retained = [r for r in records if r['disposition'] == 'retained']
        with Image.open(root / 'images/val' / f'{stem}.jpg') as handle:
            width, height = handle.size
        refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'], width, height)
        eligible = [r for r in refs if r.get('eligible', True)]

        original = [{'bbox_xyxy': r['bbox_xyxy']} for r in retained]
        updated = [{'bbox_xyxy': pick(refined_boxes.get(f'{stem}/{r["id"]}'), r['bbox_xyxy'], args.min_quality)}
                   for r in retained]
        before, after = summarise(original, eligible, args.match_iou), summarise(updated, eligible, args.match_iou)

        totals['eligible'] += len(eligible)
        totals['before'] += before['matched']
        totals['after'] += after['matched']
        totals['boxes'] += len(retained)
        totals['recovered'] += len(after['hit'] - before['hit'])
        totals['lost'] += len(before['hit'] - after['hit'])
        iou_before.append((before['mean_iou'], before['matched']))
        iou_after.append((after['mean_iou'], after['matched']))
        for index, reference in enumerate(eligible):
            was, now = index in before['hit'], index in after['hit']
            if was == now:
                continue
            bucket = by_size.setdefault(size_bin(reference['bbox_xyxy']), {'recovered': 0, 'lost': 0})
            bucket['recovered' if now else 'lost'] += 1
        rows.append({'image': stem, 'eligible': len(eligible), 'before': before['matched'],
                     'after': after['matched']})

    weighted = lambda pairs: sum(v * n for v, n in pairs) / max(sum(n for _, n in pairs), 1)
    print(f'{"image":26s} {"eligible":>8s} {"before":>7s} {"after":>6s}')
    for row in rows:
        print(f'{row["image"]:26s} {row["eligible"]:8d} {row["before"]:7d} {row["after"]:6d}')
    print(f'\nretained boxes unchanged in number: {totals["boxes"]}')
    print(f'matched {totals["before"]} -> {totals["after"]}   '
          f'recall {totals["before"] / totals["eligible"]:.4f} -> {totals["after"] / totals["eligible"]:.4f}')
    print(f'references recovered by refinement: {totals["recovered"]}')
    print(f'references lost to refinement     : {totals["lost"]}')
    print(f'mean IoU of matches {weighted(iou_before):.4f} -> {weighted(iou_after):.4f}')
    print(f'\nby reference size bin:')
    for size, counts in sorted(by_size.items()):
        print(f'  {size:8s} recovered {counts["recovered"]:3d}  lost {counts["lost"]:3d}')

    if args.out:
        Path(args.out).write_text(json.dumps({'totals': totals, 'per_image': rows, 'by_size': by_size,
                                              'mean_iou_before': weighted(iou_before),
                                              'mean_iou_after': weighted(iou_after)}, indent=1))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
