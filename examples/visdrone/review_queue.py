"""Does the review order reach the corrections sooner than the detector's score does?

A reviewer with time for k boxes out of two thousand needs the k worth looking at. Two things make
that measurable on a finite-category reference, and they have to be kept apart.

A box that matches an eligible reference loosely -- found at IoU 0.5 but not at 0.75 -- is a
correction the reference determines. It is a real object, its box is wrong, and no assumption about
unannotated objects enters. That curve is a measurement.

A box matching no reference is either a real object VisDrone omits or a false positive, so finding
those sooner is worth something to a reviewer and is not evidence of error. That curve is reported
separately and read one-sidedly, the way verifier confirmation already is.

The baseline is the detector's score ascending, which costs nothing. An order that does not beat it
has produced a more complicated way to spend the same review budget.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.evaluation import match  # noqa: E402
from lynceus.policy import load_ontology  # noqa: E402
from lynceus.review import queue  # noqa: E402
from build_benchmark import references, select_images  # noqa: E402
from risk_coverage import rebuild_evidence  # noqa: E402


def classify(objects, eligible):
    """Per box: loose match (a determined correction), tight match, or unmatched."""
    predictions = [{'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id']} for o in objects]
    loose = {index for index, _, _ in match(predictions, eligible, .5, False)}
    tight = {index for index, _, _ in match(predictions, eligible, .75, False)}
    out = []
    for index in range(len(objects)):
        if index in tight:
            out.append('tight')
        elif index in loose:
            out.append('loose')
        else:
            out.append('unmatched')
    return out


def curve(order, kinds, budgets):
    """How many of each kind the first k reviewed boxes contain."""
    rows = []
    for budget in budgets:
        seen = [kinds[index] for index in order[:budget]]
        rows.append({'reviewed': budget,
                     'loose': sum(1 for k in seen if k == 'loose'),
                     'unmatched': sum(1 for k in seen if k == 'unmatched'),
                     'tight': sum(1 for k in seen if k == 'tight')})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict-v2.json'))
    parser.add_argument('--runs', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--budgets', default='50,100,200,400,800')
    parser.add_argument('--out')
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    ontology = load_ontology(mapping.get('ontology'))
    runs = Path(args.runs)
    budgets = [int(v) for v in args.budgets.split(',')]

    ranked, scored, kinds = [], [], []
    for name in select_images(root, args.count, args.seed):
        stem = Path(name).stem
        path = runs / stem / 'annotation.json'
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        candidates = [json.loads(line) for line in
                      (runs / stem / 'evidence/candidates.jsonl').read_text().splitlines()]
        evidence = rebuild_evidence(result, candidates, ontology)
        refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'],
                             *result['image']['normalized_size'])
        eligible = [r for r in refs if r.get('eligible', True)]
        base = len(kinds)
        kinds.extend(classify(result['objects'], eligible))
        index_of = {obj['id']: position for position, obj in enumerate(result['objects'])}
        # One pooled queue over the panel, so a budget is spent where it is worth most rather than
        # being divided equally between images.
        for row in queue(result['objects'], evidence):
            ranked.append((row['difficulty'], base + index_of[row['object']]))
        for position, ev in enumerate(evidence):
            scored.append((ev['score'] if ev['score'] is not None else -1.0, base + position))

    order = [index for _, index in ranked]
    baseline = [index for _, index in sorted(scored, key=lambda pair: pair[0])]
    total = {'boxes': len(kinds),
             'loose': sum(1 for k in kinds if k == 'loose'),
             'unmatched': sum(1 for k in kinds if k == 'unmatched'),
             'tight': sum(1 for k in kinds if k == 'tight')}
    report = {'ontology': ontology['id'], 'total': total,
              'review_order': curve(order, kinds, budgets),
              'score_ascending': curve(baseline, kinds, budgets)}

    print(f"{total['boxes']} retained boxes: {total['tight']} found at IoU 0.75, "
          f"{total['loose']} found only at 0.50, {total['unmatched']} matching no reference")
    print(f'{"reviewed":>9}{"loose (order)":>15}{"loose (score)":>15}'
          f'{"unmatched (order)":>19}{"unmatched (score)":>19}')
    for mine, theirs in zip(report['review_order'], report['score_ascending']):
        print(f'{mine["reviewed"]:>9}{mine["loose"]:>15}{theirs["loose"]:>15}'
              f'{mine["unmatched"]:>19}{theirs["unmatched"]:>19}')
    print('\nLoose matches are corrections the reference determines. Unmatched boxes are not: a real '
          'object this dataset omits looks the same as a false positive.')
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f'report -> {args.out}')


if __name__ == '__main__':
    main()
