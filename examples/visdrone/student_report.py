"""Read the arms against the human arm, and say what each label decision was worth.

The human arm is the denominator because it answers the question a consumer asks: not whether the
auto-labels are correct, which a finite-category reference cannot settle, but what a model trained on
them is worth against one trained on labels a person made for the same images.

Two readings are kept separate. Down the threshold ladder, under a fixed label policy, the question
is whether precision bought by a stricter verifier threshold survives into a trained model -- the
published claim is that it does not past a moderate threshold, and this is where that is checked.
Across the label policies, at a fixed threshold, the question is what the parent fallback costs and
whether the verifier's own label is a better way to resolve it than a class prior.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, help='json written by train_students.py')
    parser.add_argument('--export', help='json written by export_students.py, for box counts')
    parser.add_argument('--out', help='write the report as json as well as printing it')
    args = parser.parse_args()

    payload = json.loads(Path(args.results).read_text())
    by_arm = defaultdict(list)
    for row in payload['runs']:
        by_arm[row['arm']].append(row)
    boxes = {}
    if args.export:
        for arm in json.loads(Path(args.export).read_text())['arms']:
            boxes[arm['arm']] = arm
    human = {metric: mean([row[metric] for row in by_arm['human']])
             for metric in ('mAP50', 'mAP50_95', 'precision', 'recall')}

    rows = []
    for arm, runs in by_arm.items():
        row = {'arm': arm, 'seeds': len(runs),
               **{metric: round(mean([r[metric] for r in runs]), 6)
                  for metric in ('mAP50', 'mAP50_95', 'precision', 'recall')}}
        if len(runs) > 1:
            row['mAP50_spread'] = round(max(r['mAP50'] for r in runs) - min(r['mAP50'] for r in runs), 6)
        if human['mAP50']:
            row['share_of_human_mAP50'] = round(row['mAP50'] / human['mAP50'], 4)
            row['share_of_human_mAP50_95'] = round(row['mAP50_95'] / human['mAP50_95'], 4)
        if arm in boxes:
            row['boxes'] = boxes[arm]['written']
            row['boxes_vs_human'] = (round(boxes[arm]['written'] / boxes['human']['written'], 4)
                                     if boxes.get('human', {}).get('written') else None)
            row['per_class_boxes'] = boxes[arm]['per_class']
        rows.append(row)
    rows.sort(key=lambda r: (-r['mAP50']))

    report = {'protocol': payload['protocol'], 'student': payload['student'],
              'human': {k: round(v, 6) for k, v in human.items()}, 'arms': rows,
              'reading': ('the share of human performance is the number the auto-labelling claim is '
                          'made in; it measures consumer utility under this dataset\'s three-class '
                          'convention and not correctness. A seed spread wider than the gap between '
                          'two arms means those two arms are not separated by this experiment.')}
    width = max(len(r['arm']) for r in rows)
    print(f'{"arm".ljust(width)}  seeds  boxes   mAP50   mAP50-95  share of human')
    for row in rows:
        print(f'{row["arm"].ljust(width)}  {row["seeds"]:5d}  {row.get("boxes", 0):6d}  '
              f'{row["mAP50"]:.4f}  {row["mAP50_95"]:.4f}    '
              f'{row.get("share_of_human_mAP50", float("nan")):.3f}')
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
