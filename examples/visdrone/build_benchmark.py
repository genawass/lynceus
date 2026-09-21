"""Build a lynceus.benchmark/1.0 payload from VisDrone2019-DET val and completed lynceus runs.

The mapping, image selection, category scope and exclusions are frozen in mapping.json and are
declared before any prediction is read. This script only applies them; it makes no choices of
its own, so it cannot be tuned against results.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image


def select_images(root, count, seed):
    files = sorted(p.name for p in (root / 'images/val').glob('*.jpg'))
    return sorted(random.Random(seed).sample(files, count))


def references(annotation_path, categories, width, height):
    """Map VisDrone rows to reference records, clipping to the source and recording every drop."""
    refs, excluded = [], {}
    for index, line in enumerate(annotation_path.read_text().splitlines()):
        parts = line.strip().split(',')
        if len(parts) < 6 or not parts[0]:
            continue
        x, y, w, h = (int(parts[i]) for i in range(4))
        category = categories[str(int(parts[5]))]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            excluded['degenerate_after_clipping'] = excluded.get('degenerate_after_clipping', 0) + 1
            continue
        record = {'id': f'ref-{index}', 'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
                  'visdrone_category': category['name']}
        if category.get('eligible') is False:
            # Kept out of every denominator, but still counted and inspectable.
            record.update(eligible=False, permitted_labels=['entity'], exclusion_reason=category['reason'])
            excluded[category['reason']] = excluded.get(category['reason'], 0) + 1
        else:
            record['permitted_labels'] = category['permitted_labels']
        refs.append(record)
    return refs, excluded


def predictions(run_dir):
    if run_dir is None or not (run_dir / 'annotation.json').is_file():
        return 'missing', []
    result = json.loads((run_dir / 'annotation.json').read_text())
    if result['execution']['status'] != 'completed':
        return result['execution']['status'], []
    return 'completed', [{'id': o['id'], 'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id'],
                          'status': o['status'], 'kind': o['kind'], 'scene_layer': o['scene_layer']}
                         for o in result['objects']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping.json'))
    parser.add_argument('--runs', help='directory holding one lynceus run directory per image stem')
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    runs = Path(args.runs) if args.runs else None
    images, totals = [], {}
    for name in select_images(root, args.count, args.seed):
        stem = Path(name).stem
        path = root / 'images/val' / name
        with Image.open(path) as im:
            width, height = im.size
        refs, excluded = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'], width, height)
        for reason, n in excluded.items():
            totals[reason] = totals.get(reason, 0) + n
        status, preds = predictions(runs / stem if runs else None)
        images.append({'id': stem, 'split': mapping['split'], 'domain': mapping['domain'],
                       'size': [width, height], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'scene_id': stem.split('_')[0], 'near_duplicate_group': stem.split('_')[0],
                       'run_status': status, 'references': refs, 'predictions': preds})

    payload = {'schema_version': 'lynceus.benchmark/1.0', 'frozen': True,
               'ontology': mapping.get('ontology', 'lynceus-broad-v1'),
               'reference_provenance': 'independent', 'policy_compatible': True,
               'reference_mapping_id': mapping['id'],
               'category_scope': mapping['category_scope'],
               'category_scope_reason': mapping['category_scope_reason'],
               'known_limitations': mapping['known_limitations'],
               'reference_exclusions': totals, 'images': images}
    Path(args.output).write_text(json.dumps(payload, indent=2))
    eligible = sum(sum(r.get('eligible', True) for r in im['references']) for im in images)
    print(f"{len(images)} images, {eligible} eligible references, exclusions {totals or '{}'}")
    print(f"runs found: {sum(im['run_status'] == 'completed' for im in images)}/{len(images)}")


if __name__ == '__main__':
    main()
