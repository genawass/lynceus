"""Risk versus coverage across acceptance thresholds (the WP-09 report).

Raising an acceptance threshold always improves precision on what survives. The gate that matters
is whether useful coverage survives with it, so both are reported at every operating point --
including the all-abstention point, where precision is undefined rather than perfect.

Thresholds are re-applied offline to the evidence already recorded in each run, so the sweep costs
no inference and cannot change what was detected.
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


def rebuild_evidence(result, candidates, ontology):
    """Recover each retained object's evidence from the run's own candidate store.

    Retained candidates are written in object order and the verification stage pairs them that way,
    so the replay pairs them that way too. Keying on geometry instead was wrong as soon as
    refinement existed: a refined object carries the refiner's box while its retained candidate
    still carries the proposal, every lookup missed, and every object replayed as an unverified
    zero-score detection. A length mismatch is therefore an error rather than something to paper
    over with a fallback.
    """
    useful = {c['id']: c['useful'] for c in ontology['classes']}
    leads = [c for c in candidates if c['disposition'] == 'retained']
    if len(leads) != len(result['objects']):
        raise SystemExit(f'{len(leads)} retained candidates for {len(result["objects"])} objects; '
                         'the run cannot be replayed')
    evidence = []
    for obj, lead in zip(result['objects'], leads):
        verification = lead.get('verification') or {}
        evidence.append({'score': lead['model_score'],
                         'views': len(lead.get('merged_views', [])) or 1,
                         'view_edge': bool(lead['view_edge']),
                         'alternatives': obj['alternative_boxes'],
                         'useful': useful.get(obj['label']['id'], False),
                         # Verification travels too, so a sweep can re-apply rules that read it.
                         'verified': verification.get('confirmed'),
                         'verifier_score': verification.get('verifier_score'),
                         'verifier_overlap': verification.get('overlap')})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping.json'))
    parser.add_argument('--runs', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scores', default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95')
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
        candidates = [json.loads(l) for l in (runs / stem / 'evidence/candidates.jsonl').read_text().splitlines()]
        refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'], *result['image']['normalized_size'])
        loaded.append((result, rebuild_evidence(result, candidates, ontology),
                       [r for r in refs if r.get('eligible', True)],
                       result['search']['mandatory_planned']))

    eligible_total = sum(len(e) for _, _, e, _ in loaded)
    rows = []
    print(f'{"score":>7}{"accepted":>10}{"correct":>9}{"precision":>11}{"coverage":>10}{"risk":>8}')
    print('-' * 55)
    for score in [float(v) for v in args.scores.split(',')]:
        rules = {**ACCEPTANCE_DEFAULTS, 'score': score}
        accepted_n = correct_n = 0
        for result, evidence, eligible, planned in loaded:
            objects = assess(copy.deepcopy(result['objects']), evidence, planned, rules)
            accepted = [{'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id']}
                        for o in objects if o['status'] == 'accepted']
            accepted_n += len(accepted)
            correct_n += len(match(accepted, eligible, .75, True))
        precision = correct_n / accepted_n if accepted_n else None
        coverage = correct_n / eligible_total
        rows.append({'score': score, 'accepted': accepted_n, 'correct': correct_n,
                     'precision': precision, 'useful_coverage': coverage})
        shown = '      -   ' if precision is None else f'{precision:11.3f}'
        risk = '      - ' if precision is None else f'{1 - precision:8.3f}'
        print(f'{score:>7.2f}{accepted_n:>10}{correct_n:>9}{shown}{coverage:>10.3f}{risk}')
    print(f'\neligible references {eligible_total}; acceptance at IoU 0.75 with a permitted useful label.')
    print('Precision on an empty accepted set is undefined, never 1.0; useful coverage is then 0.')
    if args.out:
        Path(args.out).write_text(json.dumps({'ontology': ontology['id'], 'eligible': eligible_total, 'sweep': rows}, indent=2))
        print(f'report -> {args.out}')


if __name__ == '__main__':
    main()
