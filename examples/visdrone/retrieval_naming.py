"""Can retrieval resolve a parent fallback that conservative naming refuses to resolve?

Naming is the weakest step once classes are fine-grained: on the held-out panel the pipeline emits
the parent `vehicle` on 64.7% of matched boxes, and on APOCE 96.9%. The verifier's own label is one
way to refine that and is measured elsewhere. This is the other: embed the crop, compare it against
a bank of human-labelled reference crops from the training split, and take the class its nearest
neighbours carry.

What makes this a different experiment from the verifier is what it cannot do. Retrieval emits only
classes the bank contains, so it cannot invent one, and the bank is human labels of this domain
rather than a text prompt whose phrasing is an uncontrolled variable. What makes it a fair one is
that the bank excludes every sequence the panel contains, so no crop of a panel frame can be its
own nearest neighbour.

Three namings are scored on exactly the same boxes -- exact, broader-but-permitted, or wrong --
against the reference each box matched: the pipeline's label, the verifier's label where it
confirmed the box, and retrieval over the bank.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image  # noqa: E402

from lynceus.adapters.clip import ClipEmbedder  # noqa: E402
from lynceus.evaluation import match  # noqa: E402
from lynceus.policy import load_ontology  # noqa: E402
from build_benchmark import references, select_images  # noqa: E402


def sequence(stem):
    return stem.split('_')[0]


def score(label, reference, ontology_classes):
    """Exact, broader (a parent the mapping permits), or wrong -- the label_report convention."""
    if label is None:
        return 'absent'
    permitted = reference['permitted_labels']
    if not permitted:
        return 'unscoreable'
    if label == permitted[0]:
        return 'exact'
    if label in permitted:
        return 'broader'
    return 'wrong'


def build_bank(root, categories, embedder, excluded, per_class, seed, min_side):
    """Reference crops from the training split, excluding every sequence the panel contains."""
    stems = sorted({Path(line).stem for line in (root / 'train.txt').read_text().split() if line})
    stems = [s for s in stems if sequence(s) not in excluded]
    random.Random(seed).shuffle(stems)
    wanted = {spec['permitted_labels'][0]: per_class
              for spec in categories.values() if spec.get('permitted_labels')}
    found = Counter()
    bank_rows, bank_labels, used = [], [], []
    for stem in stems:
        if all(found[label] >= wanted[label] for label in wanted):
            break
        annotation = root / 'annotations/train' / f'{stem}.txt'
        if not annotation.is_file():
            continue
        image = Image.open(root / 'images/train' / f'{stem}.jpg').convert('RGB')
        boxes, labels = [], []
        for line in annotation.read_text().splitlines():
            if not line.strip():
                continue
            values = [int(float(v)) for v in line.strip().rstrip(',').split(',')[:8]]
            x, y, w, h, _, category = values[:6]
            spec = categories.get(str(category), {})
            if not spec.get('permitted_labels'):
                continue
            label = spec['permitted_labels'][0]
            if found[label] >= wanted[label] or min(w, h) < min_side:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(label)
            found[label] += 1
        if not boxes:
            continue
        bank_rows.append(embedder.embed(image, boxes))
        bank_labels.extend(labels)
        used.append(stem)
    import torch
    return (torch.cat(bank_rows) if bank_rows else torch.zeros(0, 1)), bank_labels, used, dict(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict-v2.json'))
    parser.add_argument('--runs', required=True)
    parser.add_argument('--clip', default='/home/genadiy/data/models/clip-vit-large-patch14')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1, help='1 is the held-out panel')
    parser.add_argument('--bank-seed', type=int, default=7)
    parser.add_argument('--per-class', type=int, default=200)
    parser.add_argument('--min-side', type=int, default=8, help='bank crops smaller than this are skipped')
    parser.add_argument('--context', type=float, default=1.5)
    parser.add_argument('--neighbours', type=int, default=5)
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out')
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    ontology = load_ontology(mapping.get('ontology'))
    classes = {c['id']: c for c in ontology['classes']}
    runs = Path(args.runs)
    embedder = ClipEmbedder(args.clip, device=args.device, context=args.context)

    panel = [Path(name).stem for name in select_images(root, args.count, args.seed)]
    excluded = {sequence(stem) for stem in panel}
    bank, bank_labels, bank_images, bank_counts = build_bank(
        root, mapping['categories'], embedder, excluded, args.per_class, args.bank_seed, args.min_side)
    print(f'bank: {len(bank_labels)} crops from {len(bank_images)} training images, {bank_counts}',
          flush=True)

    tally = {'lynceus': Counter(), 'verifier': Counter(), 'retrieval': Counter()}
    fallback = {'lynceus': Counter(), 'verifier': Counter(), 'retrieval': Counter()}
    confusions = Counter()
    rows = 0
    for stem in panel:
        path = runs / stem / 'annotation.json'
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        candidates = [json.loads(line) for line in
                      (runs / stem / 'evidence/candidates.jsonl').read_text().splitlines()]
        verification = {c['verification']['object']: c['verification']
                        for c in candidates if c.get('verification')}
        refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'],
                             *result['image']['normalized_size'])
        eligible = [r for r in refs if r.get('eligible', True)]
        objects = result['objects']
        predictions = [{'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id']} for o in objects]
        pairs = match(predictions, eligible, args.match_iou, False)
        if not pairs:
            continue
        image = Image.open(root / 'images/val' / f'{stem}.jpg').convert('RGB')
        boxes = [objects[index]['bbox_xyxy'] for index, _, _ in pairs]
        retrieved = embedder.nearest(embedder.embed(image, boxes), bank, bank_labels,
                                     k=args.neighbours)
        for (index, reference_index, _), neighbour in zip(pairs, retrieved):
            obj = objects[index]
            reference = eligible[reference_index]
            emitted = obj['label']['id']
            verifier = (verification.get(obj['id']) or {}).get('verifier_label')
            readings = {'lynceus': emitted, 'verifier': verifier, 'retrieval': neighbour['label']}
            rows += 1
            for name, label in readings.items():
                outcome = score(label, reference, classes)
                tally[name][outcome] += 1
                # The population that matters is the one conservative naming declined to resolve.
                if emitted == 'vehicle':
                    fallback[name][outcome] += 1
            if emitted == 'vehicle':
                confusions[(reference['permitted_labels'][0], neighbour['label'])] += 1

    def share(counter):
        total = sum(counter.values())
        return {key: round(value / total, 4) for key, value in sorted(counter.items())} if total else {}

    report = {'ontology': ontology['id'], 'matched_boxes': rows,
              'bank': {'crops': len(bank_labels), 'images': len(bank_images),
                       'per_class': bank_counts, 'excluded_sequences': sorted(excluded),
                       'context': args.context, 'neighbours': args.neighbours},
              'all_matched': {name: share(counter) for name, counter in tally.items()},
              'parent_fallback_only': {name: share(counter) for name, counter in fallback.items()},
              'fallback_boxes': sum(fallback['lynceus'].values()),
              'retrieval_confusions_on_fallback': {f'{a}->{b}': n for (a, b), n
                                                   in confusions.most_common(12)}}
    print(f'\n{rows} matched boxes; {report["fallback_boxes"]} of them named `vehicle` by the pipeline')
    for population in ('all_matched', 'parent_fallback_only'):
        print(f'\n{population}')
        for name in ('lynceus', 'verifier', 'retrieval'):
            print(f'  {name:>10}: {report[population][name]}')
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f'\nreport -> {args.out}')


if __name__ == '__main__':
    main()
