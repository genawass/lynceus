"""Write a retrieved class per retained box, as a sidecar beside each run.

The retrieval naming measurement showed that on the held-out panel, nearest-neighbour retrieval over
a bank of human-labelled crops resolves more parent fallbacks into usable classes than the verifier
does -- 0.547 of them against 0.357 under the consumer's three-class schema -- at a higher error
rate, 0.135 against 0.036. Which trade is better is not something a naming table can settle, because
a wrong child label and an absent label cost a consumer differently. The student experiment prices
exactly that, so this writes the labels an export would use and lets it.

The bank excludes every sequence the labelled images belong to. Without that the experiment would be
reading human labels of the very frames it is labelling, which is not retrieval but leakage.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image  # noqa: E402

from lynceus.adapters.clip import ClipEmbedder  # noqa: E402
from retrieval_naming import build_bank, sequence  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default=str(Path(__file__).parent / 'student-protocol.json'))
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict-v2.json'))
    parser.add_argument('--runs', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--clip', default='/home/genadiy/data/models/clip-vit-large-patch14')
    parser.add_argument('--bank-seed', type=int, default=7)
    parser.add_argument('--per-class', type=int, default=200)
    parser.add_argument('--min-side', type=int, default=8)
    parser.add_argument('--context', type=float, default=1.5)
    parser.add_argument('--neighbours', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(protocol['reference_root'])
    runs = Path(args.runs)
    stems = sorted(p.name for p in runs.iterdir() if p.is_dir())
    excluded = {sequence(stem) for stem in stems}
    embedder = ClipEmbedder(args.clip, device=args.device, context=args.context)
    bank, bank_labels, bank_images, counts = build_bank(
        root, mapping['categories'], embedder, excluded, args.per_class, args.bank_seed,
        args.min_side)
    print(f'bank: {len(bank_labels)} crops from {len(bank_images)} images, {counts}', flush=True)
    print(f'excluded {len(excluded)} sequences the labelled runs belong to', flush=True)

    written = 0
    for index, stem in enumerate(stems, start=1):
        result = json.loads((runs / stem / 'annotation.json').read_text())
        objects = result['objects']
        if not objects:
            continue
        image = Image.open(root / f'images/{args.split}' / f'{stem}.jpg').convert('RGB')
        found = embedder.nearest(embedder.embed(image, [o['bbox_xyxy'] for o in objects]),
                                 bank, bank_labels, k=args.neighbours)
        payload = {'bank': {'crops': len(bank_labels), 'images': len(bank_images),
                            'per_class': counts, 'excluded_sequences': sorted(excluded),
                            'context': args.context, 'neighbours': args.neighbours},
                   'labels': {obj['id']: row for obj, row in zip(objects, found)}}
        (runs / stem / 'evidence/retrieval.json').write_text(json.dumps(payload, indent=1))
        written += len(found)
        if index % 25 == 0:
            print(f'[{index}/{len(stems)}] {written} boxes labelled', flush=True)
    print(f'{written} retrieved labels written beside {len(stems)} runs')


if __name__ == '__main__':
    main()
