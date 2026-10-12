"""Turn labelled runs into one training set per declared arm, and say what each arm dropped.

This is WP-11's export in the one form the experiment needs: a consumer's training set. Every arm
sees the same images and differs only in labels, so anything that moves between arms is the labels.

Two things are deliberately explicit. A box the verifier did not confirm is not written, and a box
whose label is the parent `vehicle` is resolved by the arm's declared policy rather than by a choice
made here -- conservative naming emits that parent on most matched boxes, so how it is resolved is a
label policy with a measurable cost. And what the export cannot carry travels in the sidecar: this
pipeline records unresolved and unsearched regions, a trainer has no ignore semantics, so those
regions train as background. Counting them is the difference between a known limitation and a
silent one.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

CLASSES = ('person', 'car', 'longvehicle')


def to_yolo(box, width, height, index):
    x1, y1, x2, y2 = box
    return (f'{index} {((x1 + x2) / 2) / width:.6f} {((y1 + y2) / 2) / height:.6f} '
            f'{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}')


def arm_label(obj, verification, mapping, policy, retrieved=None):
    """The dataset class this object trains as, or None with the reason it was dropped."""
    label = obj['label']['id']
    if label in mapping:
        return mapping[label], None
    if label != 'vehicle':
        return None, f'unmapped_class:{label}'
    if policy == 'drop':
        return None, 'parent_fallback_dropped'
    if policy == 'modal':
        return mapping['car'], None
    if policy == 'retrieval':
        label = (retrieved or {}).get('label')
        if label in mapping:
            return mapping[label], None
        return None, 'parent_fallback_unresolved_by_retrieval'
    verifier = (verification or {}).get('verifier_label')
    if verifier in mapping:
        return mapping[verifier], None
    return None, 'parent_fallback_unresolved_by_verifier'


def build(runs, arm, protocol, out):
    mapping = {k: CLASSES.index(v) for k, v in protocol['class_mapping']['ontology_to_dataset'].items()}
    root = Path(protocol['reference_root'])
    images = out / 'images/train'
    labels = out / 'labels/train'
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    stats = {'arm': arm['id'], 'images': 0, 'written': 0, 'dropped': {}, 'per_class': {},
             'unresolved_boundary': 0, 'unsearched_regions': 0, 'empty_images': 0}
    for run in sorted(p for p in runs.iterdir() if p.is_dir()):
        annotation = json.loads((run / 'annotation.json').read_text())
        width, height = annotation['image']['normalized_size']
        retrieval = {}
        sidecar = run / 'evidence/retrieval.json'
        if arm['policy'] == 'retrieval':
            if not sidecar.is_file():
                raise SystemExit(f'{run.name} has no retrieval sidecar; run retrieval_labels.py first')
            retrieval = json.loads(sidecar.read_text())['labels']
        verifications = {}
        for line in (run / 'evidence/candidates.jsonl').read_text().splitlines():
            record = json.loads(line)
            if record.get('verification'):
                verifications[record['verification']['object']] = record['verification']
        lines = []
        for obj in annotation['objects']:
            verification = verifications.get(obj['id'])
            if arm['verify_threshold'] is not None:
                if not (verification or {}).get('confirmed'):
                    stats['dropped']['unconfirmed'] = stats['dropped'].get('unconfirmed', 0) + 1
                    continue
                if verification['verifier_score'] < arm['verify_threshold']:
                    stats['dropped']['below_threshold'] = stats['dropped'].get('below_threshold', 0) + 1
                    continue
            index, reason = arm_label(obj, verification, mapping, arm['policy'],
                                      retrieval.get(obj['id']))
            if index is None:
                stats['dropped'][reason] = stats['dropped'].get(reason, 0) + 1
                continue
            lines.append(to_yolo(obj['bbox_xyxy'], width, height, index))
            stats['per_class'][CLASSES[index]] = stats['per_class'].get(CLASSES[index], 0) + 1
            if obj['uncertainty']['boundary'] != 'supported':
                stats['unresolved_boundary'] += 1
        stats['unsearched_regions'] += sum(1 for r in annotation['regions'] if r['kind'] == 'unsearched')
        stats['images'] += 1
        stats['written'] += len(lines)
        if not lines:
            stats['empty_images'] += 1
        (labels / f'{run.name}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
        link = images / f'{run.name}.jpg'
        if not link.exists():
            link.symlink_to(root / 'images/train' / f'{run.name}.jpg')
    return stats


def build_human(runs, protocol, out):
    """The same images with the dataset's own labels, which is the arm everything is read against."""
    root = Path(protocol['reference_root'])
    images = out / 'images/train'
    labels = out / 'labels/train'
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    stats = {'arm': 'human', 'images': 0, 'written': 0, 'dropped': {}, 'per_class': {},
             'unresolved_boundary': 0, 'unsearched_regions': 0, 'empty_images': 0}
    for run in sorted(p for p in runs.iterdir() if p.is_dir()):
        text = (root / 'labels/train' / f'{run.name}.txt').read_text()
        rows = [line for line in text.splitlines() if line.strip()]
        for row in rows:
            name = CLASSES[int(row.split()[0])]
            stats['per_class'][name] = stats['per_class'].get(name, 0) + 1
        stats['images'] += 1
        stats['written'] += len(rows)
        if not rows:
            stats['empty_images'] += 1
        (labels / f'{run.name}.txt').write_text('\n'.join(rows) + ('\n' if rows else ''))
        link = images / f'{run.name}.jpg'
        if not link.exists():
            link.symlink_to(root / 'images/train' / f'{run.name}.jpg')
    return stats


def write_yaml(out, protocol):
    root = Path(protocol['reference_root'])
    names = '\n'.join(f'  {i}: {name}' for i, name in enumerate(CLASSES))
    (out / 'dataset.yaml').write_text(
        f'path: {out.resolve()}\ntrain: images/train\nval: {root / "val.txt"}\n'
        f'nc: {len(CLASSES)}\nnames:\n{names}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default=str(Path(__file__).parent / 'student-protocol.json'))
    parser.add_argument('--runs', required=True, help='directory of labelled runs, one per image')
    parser.add_argument('--out', required=True, help='directory to hold one training set per arm')
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    runs = Path(args.runs)
    out = Path(args.out)
    report = []
    for arm in protocol['arms']:
        target = out / arm['id']
        stats = (build_human(runs, protocol, target) if arm['id'] == 'human'
                 else build(runs, arm, protocol, target))
        write_yaml(target, protocol)
        report.append(stats)
        print(f"{arm['id']}: {stats['written']} boxes over {stats['images']} images, "
              f"{stats['per_class']}, dropped {stats['dropped']}", flush=True)
    (out / 'export.json').write_text(json.dumps({'protocol': protocol['id'], 'arms': report}, indent=1))


if __name__ == '__main__':
    main()
