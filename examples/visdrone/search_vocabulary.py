"""Search a prompt vocabulary for OWLv2 on the development panel, and freeze the winner.

Phrasing is configuration, and configuration chosen by hand is chosen badly: nobody knows whether
this detector finds more small overhead figures under `person`, `pedestrian` or `person seen from
above` without trying them.

The search runs on the development panel only. Selecting phrasing against the held-out panel is
dataset-level prompt optimization on target data, which the constraints forbid and which would
invalidate that panel, so the winner is written out frozen and measured there exactly once,
separately, by an ordinary run.

It is affordable because a prompted detector's image features and box predictions do not depend on
the text. They are computed once per view and cached; each candidate then costs a text-tower pass
and one matrix multiply. That is a cost reduction, not an approximation -- the scores are the ones
a full forward pass gives.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.geometry import plan_tiles  # noqa: E402
from lynceus.policy import available_ontologies  # noqa: E402
from lynceus.prompt_search import coordinate_ascent, objective, score_vocabulary  # noqa: E402
from build_benchmark import references, select_images  # noqa: E402

# Enumerated in advance and recorded with the result, so a vocabulary that beat two alternatives is
# distinguishable from one that beat forty. Aerial phrasings dominate because the domain does: an
# object here is seen from above, small, and often only identifiable by its context.
CANDIDATES = {
    'person': ['person', 'pedestrian', 'a person walking', 'person seen from above',
               'aerial view of a person'],
    'vehicle': ['vehicle', 'a vehicle', 'vehicle seen from above', 'aerial view of a vehicle'],
    'car': ['car', 'a car', 'car seen from above', 'aerial view of a car', 'parked car'],
    'van': ['van', 'a van', 'delivery van', 'van seen from above'],
    'truck': ['truck', 'a truck', 'lorry', 'truck seen from above'],
    'bus': ['bus', 'a bus', 'city bus', 'bus seen from above'],
    'motorcycle': ['motorcycle', 'a motorcycle', 'motorbike', 'scooter',
                   'motorcycle seen from above'],
    'bicycle': ['bicycle', 'a bicycle', 'bike', 'cyclist', 'bicycle seen from above'],
    'tricycle': ['tricycle', 'a tricycle', 'three wheeled vehicle', 'rickshaw'],
}


def cache_views(model, processor, root, stems, mapping, tile_levels, threshold, device):
    """Image features and boxes per view, which no phrasing can change.

    The class head is left as a closure over the cached features: calling it with a set of query
    embeddings produces exactly the logits a full forward pass would, at the cost of one matrix
    multiply.
    """
    views, refs = [], {}
    for stem in stems:
        with Image.open(root / 'images/val' / f'{stem}.jpg') as handle:
            image = handle.convert('RGB')
            width, height = image.size
            jobs = plan_tiles(width, height, levels=tile_levels, overlap=0.25)
            for job in jobs:
                x1, y1, x2, y2 = (int(round(v)) for v in job['bbox_xyxy'])
                view = image if job['id'] == 'full' else image.crop((x1, y1, x2, y2))
                inputs = processor(text=[['x']], images=view, return_tensors='pt').to(device)
                with torch.no_grad():
                    feats, _ = model.image_embedder(pixel_values=inputs['pixel_values'])
                    b, h, w, d = feats.shape
                    image_feats = feats.reshape(b, h * w, d)
                    boxes = model.box_predictor(image_feats, feature_map=feats)
                side = max(view.height, view.width)
                from transformers.image_transforms import center_to_corners_format
                corners = (center_to_corners_format(boxes[0]) * side).cpu().numpy()
                corners[:, 0] = np.clip(corners[:, 0], 0, view.width) + x1
                corners[:, 2] = np.clip(corners[:, 2], 0, view.width) + x1
                corners[:, 1] = np.clip(corners[:, 1], 0, view.height) + y1
                corners[:, 3] = np.clip(corners[:, 3], 0, view.height) + y1

                def predict(query, image_feats=image_feats):
                    with torch.no_grad():
                        logits, _ = model.class_predictor(image_feats, query)
                    return torch.sigmoid(logits[0]).float().cpu().numpy()

                views.append({'image': stem, 'boxes': corners, 'predict': predict})
        rows, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'], width, height)
        refs[stem] = [r for r in rows if r.get('eligible', True)]
        print(f'  cached {stem}: {len(jobs)} views, {len(refs[stem])} references', flush=True)
    return views, refs


def embed_text(model, processor, phrases, device):
    """Query embeddings for one vocabulary, in the space the class head consumes.

    `image_text_embedder` returns the text embeddings first; the pixel values it also needs are a
    throwaway, since nothing about the queries depends on them.
    """
    inputs = processor(text=[phrases], images=Image.new('RGB', (32, 32)), return_tensors='pt').to(device)
    with torch.no_grad():
        query, _, _ = model.image_text_embedder(input_ids=inputs['input_ids'],
                                                pixel_values=inputs['pixel_values'],
                                                attention_mask=inputs['attention_mask'])
    return query.reshape(1, len(phrases), query.shape[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict.json'))
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', required=True, help='vocabulary artifact to write')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0, help='0 is the development panel; never use 1')
    parser.add_argument('--tile-levels', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--id', default='owlv2-aerial-searched-v1')
    args = parser.parse_args()

    if args.seed == 1:
        raise SystemExit('seed 1 is the held-out panel; fitting a vocabulary on it invalidates it')

    from lynceus.bundle import preflight
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    ontology = available_ontologies()[mapping['ontology']]
    classes = [c['id'] for c in ontology['classes'] if c['useful']]
    bundle = preflight(args.bundle)

    processor = Owlv2Processor.from_pretrained(bundle['model_dir'], local_files_only=True)
    model = Owlv2ForObjectDetection.from_pretrained(bundle['model_dir'], local_files_only=True).eval().to(args.device)

    stems = [Path(n).stem for n in select_images(root, args.count, args.seed)]
    print(f'caching features for the development panel (seed {args.seed})')
    views, refs = cache_views(model, processor, root, stems, mapping,
                              args.tile_levels, args.threshold, args.device)
    eligible = sum(len(v) for v in refs.values())
    print(f'{len(views)} views cached, {eligible} eligible references\n')

    seen = {}

    def evaluate(vocabulary):
        key = tuple(vocabulary[c] for c in classes)
        if key in seen:
            return seen[key]
        query = embed_text(model, processor, list(key), args.device)
        report = score_vocabulary(views, query, classes, refs, args.threshold)
        seen[key] = objective(report, eligible)
        return seen[key]

    def log(label, score, adopted=False):
        print(f'{label:52s} {score:.5f}{"  <- adopted" if adopted else ""}', flush=True)

    baseline = {c: next(x['name'] for x in ontology['classes'] if x['id'] == c) for c in classes}
    result = coordinate_ascent(classes, CANDIDATES, evaluate, baseline, args.rounds, log)

    artifact = {
        'id': args.id, 'version': '1.0', 'adapter': 'owlv2', 'ontology': ontology['id'],
        'derivation': ('Coordinate ascent over per-class phrasings, scored on the whole vocabulary '
                       'because the emitted label is the class that wins over all prompts. Cached '
                       'image features make each candidate exact and cheap.'),
        'fitted_on': {'panel': 'VisDrone2019-DET val', 'seed': args.seed, 'images': len(stems),
                      'eligible_references': eligible, 'split': 'development',
                      'tile_levels': args.tile_levels, 'threshold': args.threshold,
                      'date': date.today().isoformat()},
        'objective': 'harmonic mean of localized recall and precision over the development panel',
        'baseline_score': result['trace'][0]['score'],
        'development_score': result['score'],
        'candidates': CANDIDATES,
        'search_trace': result['trace'],
        'prompts': result['vocabulary'],
    }
    Path(args.out).write_text(json.dumps(artifact, indent=1))
    print(f"\nbaseline {artifact['baseline_score']:.5f} -> searched {artifact['development_score']:.5f}"
          f"  (+{artifact['development_score'] - artifact['baseline_score']:.5f} on development)")
    for class_id in classes:
        if result['vocabulary'][class_id] != baseline[class_id]:
            print(f"  {class_id:12s} {baseline[class_id]!r} -> {result['vocabulary'][class_id]!r}")
    print(f'\nwrote {args.out}')
    print('this is a development score; measure the frozen vocabulary on the held-out panel separately')


if __name__ == '__main__':
    main()
