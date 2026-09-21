"""Per-image class-neutral recall and precision, classification error and localization error.

Matching is one-to-one at IoU >= 0.5 ignoring labels (class neutral), the same rule the evaluator
uses for localized_instance_recall. Class and localization error are then measured only on those
matched pairs, so a localization failure cannot masquerade as a naming failure.

Precision is reported twice because VisDrone is a finite-category reference: `prec_all` charges
every retained instance, including correct detections of classes VisDrone never annotates;
`prec_sc` counts only predictions inside the declared category scope. The truth lies between.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.evaluation import match  # noqa: E402
from build_benchmark import references, select_images  # noqa: E402


def measure(stem, root, run_dir, mapping):
    refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'], 1360, 765)
    eligible = [r for r in refs if r.get('eligible', True)]
    result = json.loads((run_dir / stem / 'annotation.json').read_text())
    size = result['image']['normalized_size']
    refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'], *size)
    eligible = [r for r in refs if r.get('eligible', True)]
    preds = [{'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id']} for o in result['objects']]
    scope = set(mapping['category_scope'])
    scoped = [p for p in preds if p['label'] in scope or p['label'] in ('unknown_object', 'entity')]

    pairs = match(preds, eligible, .5)
    pairs_scoped = match(scoped, eligible, .5)
    ious = [v for _, _, v in pairs]
    misclassified = sum(preds[a]['label'] not in eligible[b]['permitted_labels'] for a, b, _ in pairs)
    loose = sum(1 for v in ious if v < .75)
    return {
        'image': stem, 'gt': len(eligible), 'pred': len(preds), 'pred_sc': len(scoped),
        'tp': len(pairs),
        'recall': len(pairs) / len(eligible) if eligible else None,
        'prec_all': len(pairs) / len(preds) if preds else None,
        'prec_sc': len(pairs_scoped) / len(scoped) if scoped else None,
        'cls_err': misclassified / len(pairs) if pairs else None,
        'loc_err': sum(1 - v for v in ious) / len(ious) if ious else None,
        'iou_lt_75': loose / len(ious) if ious else None,
        'mean_iou': sum(ious) / len(ious) if ious else None,
    }


def table(name, rows):
    head = f'{"image":<26}{"gt":>5}{"pred":>6}{"TP":>5}{"recall":>9}{"prec_all":>10}{"prec_sc":>9}{"cls_err":>9}{"loc_err":>9}{"IoU<.75":>9}'
    print(f'\n{name}\n{head}\n{"-" * len(head)}')

    def fmt(v):
        return '     -   ' if v is None else f'{v:9.3f}'
    for r in rows:
        print(f'{r["image"]:<26}{r["gt"]:>5}{r["pred"]:>6}{r["tp"]:>5}'
              f'{fmt(r["recall"])}{fmt(r["prec_all"])}{fmt(r["prec_sc"])}{fmt(r["cls_err"])}{fmt(r["loc_err"])}{fmt(r["iou_lt_75"])}')
    gt = sum(r['gt'] for r in rows); pred = sum(r['pred'] for r in rows)
    pred_sc = sum(r['pred_sc'] for r in rows); tp = sum(r['tp'] for r in rows)
    # Micro aggregates: recomputed from summed counts, never an average of per-image rates.
    weighted = lambda key: sum(r[key] * r['tp'] for r in rows if r[key] is not None) / tp if tp else None
    print(f'{"-" * len(head)}\n{"MICRO TOTAL":<26}{gt:>5}{pred:>6}{tp:>5}'
          f'{fmt(tp / gt)}{fmt(tp / pred)}{fmt(sum(r["tp"] for r in rows) / pred_sc)}'
          f'{fmt(weighted("cls_err"))}{fmt(weighted("loc_err"))}{fmt(weighted("iou_lt_75"))}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping.json'))
    parser.add_argument('--runs', action='append', required=True, metavar='NAME=DIR')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out')
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    report = {}
    for spec in args.runs:
        name, _, directory = spec.partition('=')
        rows = [measure(Path(n).stem, root, Path(directory), mapping)
                for n in select_images(root, args.count, args.seed)
                if (Path(directory) / Path(n).stem / 'annotation.json').is_file()]
        table(name, rows)
        report[name] = rows
    print('\nclass neutral: matching ignores labels. cls_err = share of matched pairs whose label is '
          'not permitted for that reference. loc_err = mean (1 - IoU) over matched pairs.')
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f'report -> {args.out}')


if __name__ == '__main__':
    main()
