"""What the verifier's own geometry adds, re-applied to evidence already recorded.

Boundary support currently rests on this detector agreeing with itself: the alternatives it is
checked against come from the detector's other views of the same object. Where a verifier confirmed
a box, the overlap that confirmed it is an independent reading of the same extent, and by the
argument that already makes multi-view disagreement boundary uncertainty, disagreement there is too.

The rule can only withhold support. It never moves a box, because refinement is the stage entitled
to change geometry and it does so against pixels rather than against a vote. So the question this
measures is not whether boxes get better -- none change -- but whether the boxes it stops accepting
are worse than the ones it keeps. If they are, the accepted set gets tighter for the coverage it
gives up; if the rule removes matched and unmatched boxes at the same rate, it is a random pruner
wearing a reason, which is the same bar `pruner selectivity` sets for any other remover.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.evaluation import match  # noqa: E402
from lynceus.pipeline import assess, ACCEPTANCE_DEFAULTS  # noqa: E402
from lynceus.policy import load_ontology  # noqa: E402
from build_benchmark import references, select_images  # noqa: E402
from risk_coverage import rebuild_evidence  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict-v2.json'))
    parser.add_argument('--runs', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--score', type=float, default=0.1, help='acceptance score, held fixed')
    parser.add_argument('--min-views', type=int, default=1)
    parser.add_argument('--overlaps', default='none,0.5,0.6,0.7,0.8,0.9')
    parser.add_argument('--out')
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    ontology = load_ontology(mapping.get('ontology'))
    runs = Path(args.runs)

    loaded = []
    for name in select_images(root, args.count, args.seed):
        stem = Path(name).stem
        path = runs / stem / 'annotation.json'
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        candidates = [json.loads(line) for line in
                      (runs / stem / 'evidence/candidates.jsonl').read_text().splitlines()]
        refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'],
                             *result['image']['normalized_size'])
        loaded.append((result, rebuild_evidence(result, candidates, ontology),
                       [r for r in refs if r.get('eligible', True)],
                       result['search']['mandatory_planned']))
    eligible_total = sum(len(e) for _, _, e, _ in loaded)

    def operating_point(rules):
        accepted_n = 0
        found = {0.5: 0, 0.75: 0, 0.85: 0}
        for result, evidence, eligible, planned in loaded:
            objects = assess(copy.deepcopy(result['objects']), evidence, planned, rules)
            accepted = [{'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id']}
                        for o in objects if o['status'] == 'accepted']
            accepted_n += len(accepted)
            for iou_threshold in found:
                found[iou_threshold] += len(match(accepted, eligible, iou_threshold, True))
        return accepted_n, found

    # The baseline any remover has to beat is the detector's own score removing the same number of
    # boxes. A rule that removes matched boxes at the score's rate has selected nothing; it has
    # only shrunk the accepted set, which always raises precision on what survives.
    baseline = []
    for score in [args.score + i / 200 for i in range(0, 100)]:
        accepted_n, found = operating_point({**ACCEPTANCE_DEFAULTS, 'score': score,
                                             'min_views': args.min_views,
                                             'require_verification': True,
                                             'verifier_boundary_iou': None})
        baseline.append({'score': round(score, 4), 'accepted': accepted_n, 'matched': found})
        if accepted_n == 0:
            break

    def nearest_baseline(accepted_n):
        return min(baseline, key=lambda row: abs(row['accepted'] - accepted_n))

    rows = []
    print(f'{"overlap":>8}{"accepted":>10}{"at .50":>8}{"at .75":>8}{"at .85":>8}'
          f'{"precision":>11}{"coverage":>10}{"score baseline at same size":>30}')
    print('-' * 63)
    for value in args.overlaps.split(','):
        threshold = None if value == 'none' else float(value)
        rules = {**ACCEPTANCE_DEFAULTS, 'score': args.score, 'min_views': args.min_views,
                 'require_verification': True, 'verifier_boundary_iou': threshold}
        accepted_n, found = operating_point(rules)
        precision = found[0.75] / accepted_n if accepted_n else None
        rival = nearest_baseline(accepted_n)
        rows.append({'verifier_boundary_iou': threshold, 'accepted': accepted_n,
                     'matched': {str(k): v for k, v in found.items()},
                     'precision_at_0.75': precision,
                     'useful_coverage': found[0.75] / eligible_total if eligible_total else None,
                     'score_baseline': {'score': rival['score'], 'accepted': rival['accepted'],
                                        'matched_at_0.75': rival['matched'][0.75],
                                        'matched_at_0.85': rival['matched'][0.85]}})
        shown = '      -   ' if precision is None else f'{precision:11.4f}'
        print(f'{value:>8}{accepted_n:>10}{found[0.5]:>8}{found[0.75]:>8}{found[0.85]:>8}{shown}'
              f'{found[0.75] / eligible_total:>10.4f}'
              f'   score {rival["score"]:.3f}: {rival["accepted"]} kept, '
              f'{rival["matched"][0.75]} at .75')
    print(f'\neligible references {eligible_total}; acceptance requires confirmation at every row, '
          'so only the extent rule varies.')
    print('A row that loses matched boxes as fast as it loses accepted boxes has selected nothing.')
    if args.out:
        Path(args.out).write_text(json.dumps(
            {'ontology': ontology['id'], 'eligible': eligible_total, 'score': args.score,
             'min_views': args.min_views, 'sweep': rows}, indent=1))
        print(f'report -> {args.out}')


if __name__ == '__main__':
    main()
