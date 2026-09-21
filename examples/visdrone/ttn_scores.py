"""Score a panel's retained boxes with TTN, for the selectivity curve.

TTN is not a standalone scorer. It is an in-context comparator: for each class it is shown ten
ground-truth reference patch embeddings of that class followed by the candidate, and the
candidate's accept/reject logit is read from the last sequence position. The adaptation is the
reference set, not the weights, so nothing here trains or fine-tunes anything.

That reference set is human labels from the target domain, which places this squarely under A3 --
allowed outside runtime, with target data excluded. Two controls follow, and both are enforced
here rather than assumed:

* references come from the train split, never the evaluation panel;
* the 24 sequences appearing in both splits are dropped, because VisDrone frames of one sequence
  are near duplicates and the mapping already treats the sequence prefix as the scene identity.
  One of those sequences is in the held-out panel, so skipping this check would leak directly.

TTN's own code is imported from a local clone rather than vendored: the repository states no
licence, which blocks redistribution though not measurement.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_benchmark import select_images  # noqa: E402

MIN_REFERENCES = 10          # nref in the shipped model.yaml; a class below this cannot be scored
ABSTAIN = ('unknown_object', 'entity')


def sequence(stem):
    return stem.split('_')[0]


def crop_embeddings(model, processor, image, boxes, device, batch=64):
    """CLIP ViT-L-14 label-patch embeddings, float16, matching what TTN was trained to consume."""
    out = []
    for start in range(0, len(boxes), batch):
        patches = []
        for box in boxes[start:start + batch]:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
            patches.append(image.crop((x1, y1, x2, y2)))
        inputs = processor(images=patches, return_tensors='pt').to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        out.append(features.to(torch.float16).cpu())
    return torch.cat(out) if out else torch.zeros((0, 768), dtype=torch.float16)


def reference_bank(root, mapping, model, processor, device, per_class, seed, exclude):
    """Up to `per_class` leak-free ground-truth patch embeddings for each ontology class."""
    specific = {k: (v['permitted_labels'][0] if v.get('eligible', True) else None)
                for k, v in mapping['categories'].items()}
    files = [f for f in sorted((root / 'annotations/train').glob('*.txt'))
             if sequence(f.stem) not in exclude]
    random.Random(seed).shuffle(files)
    wanted = {c for c in specific.values() if c}
    bank = {c: [] for c in wanted}
    for path in files:
        if all(len(v) >= per_class for v in bank.values()):
            break
        rows = []
        for line in path.read_text().splitlines():
            parts = line.strip().split(',')
            if len(parts) < 6 or not parts[0]:
                continue
            label = specific.get(str(int(parts[5])))
            if label is None or len(bank[label]) >= per_class:
                continue
            x, y, w, h = (int(parts[i]) for i in range(4))
            if w <= 0 or h <= 0:
                continue
            rows.append((label, [x, y, x + w, y + h]))
        if not rows:
            continue
        with Image.open(root / 'images/train' / f'{path.stem}.jpg') as handle:
            image = handle.convert('RGB')
            embeddings = crop_embeddings(model, processor, image, [b for _, b in rows], device)
        for (label, _), embedding in zip(rows, embeddings):
            if len(bank[label]) < per_class:
                bank[label].append(embedding)
    return {c: torch.stack(v) for c, v in bank.items() if v}


def score_candidates(ttn, embeddings, references, device, batch=500):
    """Mean accept/reject logit over every chunk of ten references, as the paper's script does.

    The sequence the model expects is the ten reference embeddings followed by the candidate, and
    the candidate's logit is the last position. That layout is built here rather than imported:
    the repository's helper pulls in FiftyOne through an unused import, and the licensed artifact
    is the checkpoint, not a batching loop.
    """
    total = torch.zeros(len(embeddings))
    chunks = 0
    for start in range(0, len(references) - MIN_REFERENCES + 1, MIN_REFERENCES):
        block = references[start:start + MIN_REFERENCES].to(device)
        for index in range(0, len(embeddings), batch):
            candidates = embeddings[index:index + batch].to(device).unsqueeze(1)
            window = torch.cat((block.unsqueeze(0).repeat(len(candidates), 1, 1), candidates), 1)
            with torch.no_grad():
                logits, _ = ttn(window)
            total[index:index + batch] += logits[:, -1].float().cpu()
        chunks += 1
    return (total / chunks) if chunks else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict.json'))
    parser.add_argument('--runs', required=True, help='shipped run directory tree')
    parser.add_argument('--ttn-repo', required=True, help='local clone of github.com/voxel51/ttn')
    parser.add_argument('--ttn-weights', required=True, help='directory holding best.pth and model.yaml')
    parser.add_argument('--clip', default='/home/genadiy/data/models/clip-vit-large-patch14')
    parser.add_argument('--refs-per-class', type=int, default=100)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.ttn_repo) / 'ttn'))
    from ttn_model import load_ttn_model
    from transformers import CLIPImageProcessor, CLIPModel

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    device = args.device

    train_sequences = {sequence(p.stem) for p in (root / 'images/train').glob('*.jpg')}
    val_sequences = {sequence(p.stem) for p in (root / 'images/val').glob('*.jpg')}
    shared = train_sequences & val_sequences
    stems = [Path(n).stem for n in select_images(root, args.count, args.seed)]
    leaking = sorted({sequence(s) for s in stems} & shared)
    print(f'{len(shared)} sequences appear in both splits and are excluded from the reference bank')
    print(f'panel sequences that would have leaked: {leaking or "none"}')

    clip = CLIPModel.from_pretrained(args.clip, local_files_only=True).eval().to(device)
    processor = CLIPImageProcessor.from_pretrained(args.clip, local_files_only=True)

    print(f'building reference bank, up to {args.refs_per_class} patches per class')
    bank = reference_bank(root, mapping, clip, processor, device, args.refs_per_class, args.seed, shared)
    for label, embeddings in sorted(bank.items()):
        print(f'  {label:12s} {len(embeddings):4d}')
    bank['vehicle'] = torch.cat([v for k, v in bank.items() if k != 'person'])[:args.refs_per_class]
    print(f'  {"vehicle":12s} {len(bank["vehicle"]):4d}  (union of the specific vehicle classes)')

    ttn = load_ttn_model(args.ttn_weights, 768, device)

    scores, unscored, scored = {}, 0, 0
    for stem in stems:
        records = [json.loads(line) for line
                   in (Path(args.runs) / stem / 'evidence/candidates.jsonl').read_text().splitlines() if line]
        retained = [r for r in records if r['disposition'] == 'retained']
        with Image.open(root / 'images/val' / f'{stem}.jpg') as handle:
            image = handle.convert('RGB')
            embeddings = crop_embeddings(clip, processor, image, [r['bbox_xyxy'] for r in retained], device)
        by_label = {}
        for index, record in enumerate(retained):
            by_label.setdefault(record['label'], []).append(index)
        for label, indices in by_label.items():
            key = [f'{stem}/{retained[i]["id"]}' for i in indices]
            references = bank.get(label)
            # An abstained box makes no class claim, so there is no class reference set to compare
            # it against. Scoring it against some other class's references would invent a decision.
            if label in ABSTAIN or references is None or len(references) < MIN_REFERENCES:
                for k in key:
                    scores[k] = None
                unscored += len(key)
                continue
            logits = score_candidates(ttn, embeddings[indices], references, device)
            for k, value in zip(key, logits.tolist()):
                scores[k] = float(value)
            scored += len(key)
        print(f'  {stem}: {len(retained)} retained', flush=True)

    Path(args.out).write_text(json.dumps(scores, indent=1))
    print(f'\nscored {scored}, unscored {unscored} (abstained or no class reference set)')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
