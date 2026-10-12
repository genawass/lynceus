"""Verify a panel's retained boxes against any bundled verifier, and emit comparable rows.

The verifier is selected by the adapter its bundle declares, so SAM 3 and Grounding DINO produce
the same row schema and can be compared directly. That comparison is the reason a second verifier
exists: a third architecture is only worth its cost if it confirms *different* boxes, and two
verifiers agreeing on the same set add nothing but runtime.

What each run supports is still one-sided. Sensitivity is measurable because boxes matching an
eligible reference are known positives; specificity is not, because no box here is a known
negative. So a low confirmation rate among undecided boxes remains consistent both with those
boxes being false and with the verifier missing them.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image  # noqa: E402

from lynceus.bundle import preflight  # noqa: E402
from lynceus.evaluation import match  # noqa: E402
from lynceus.geometry import iou  # noqa: E402
from lynceus.policy import load_ontology  # noqa: E402
from lynceus.vocabulary import check_adapter, load_vocabulary, resolve_prompts  # noqa: E402
from build_benchmark import references, select_images  # noqa: E402

SIZE_BINS = ((16, '<16'), (32, '16-32'), (96, '32-96'))


def size_bin(box):
    side = ((box[2] - box[0]) * (box[3] - box[1])) ** .5
    for limit, name in SIZE_BINS:
        if side < limit:
            return name
    return '>=96'


def build_verifier(bundle, device, tile_side, threshold):
    adapter = preflight(bundle)['adapter']
    if adapter == 'sam3':
        from lynceus.adapters.sam3 import Sam3Verifier
        return adapter, Sam3Verifier(bundle, device=device, tile_side=tile_side, threshold=threshold)
    if adapter == 'grounding-dino':
        from lynceus.adapters.grounding_dino import GroundingDinoVerifier
        return adapter, GroundingDinoVerifier(bundle, device=device, tile_side=tile_side, threshold=threshold)
    if adapter == 'wedetect-uni':
        from lynceus.adapters.wedetect import WeDetectProposer
        # tile_side 0 means one pass over the whole frame, which its 1280 input is built for.
        return adapter, WeDetectProposer(bundle, device=device,
                                         tile_side=tile_side or None, threshold=threshold)
    raise SystemExit(f'no verifier for adapter {adapter}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict-v2.json'))
    parser.add_argument('--runs', required=True)
    parser.add_argument('--verify-bundle', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--vocabulary')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--tile-side', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    ontology = load_ontology(mapping['ontology'])
    vocabulary = load_vocabulary(args.vocabulary) if args.vocabulary else None
    adapter, verifier = build_verifier(args.verify_bundle, args.device, args.tile_side, args.threshold)
    if adapter == 'wedetect-uni':
        # A class-agnostic proposer is asked nothing, so there is no vocabulary to resolve.
        prompts = {}
    else:
        resolution = check_adapter(resolve_prompts(ontology, vocabulary), adapter)
        prompts = {phrase: class_id for class_id, phrase in zip(resolution['classes'], resolution['prompts'])}

    rows, started = [], time.perf_counter()
    for index, name in enumerate(select_images(root, args.count, args.seed), start=1):
        stem = Path(name).stem
        result = json.loads((Path(args.runs) / stem / 'annotation.json').read_text())
        retained = [{'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id'], 'id': o['id']}
                    for o in result['objects']]
        refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'],
                             *result['image']['normalized_size'])
        eligible = [r for r in refs if r.get('eligible', True)]
        known = {a for a, _, _ in match(retained, eligible, .5)}

        image = Image.open(root / 'images/val' / f'{stem}.jpg').convert('RGB')
        detections = verifier.detect(image, prompts)
        for position, box in enumerate(retained):
            best, best_iou = None, 0.0
            for d in detections:
                value = iou(box['bbox_xyxy'], d['bbox_xyxy'])
                if value >= args.match_iou and value > best_iou:
                    best, best_iou = d, value
            rows.append({'image': stem, 'id': box['id'], 'label': box['label'],
                         'size_bin': size_bin(box['bbox_xyxy']),
                         'known': 'true_positive' if position in known else 'undecided',
                         'confirmed': best is not None,
                         'verifier_label': best['label'] if best else None,
                         'verifier_score': round(best['score'], 6) if best else None})
        confirmed = sum(1 for r in rows[-len(retained):] if r['confirmed'])
        print(f'[{index}/{args.count}] {stem}: {len(retained)} boxes, {len(detections)} detections, '
              f'{confirmed} confirmed', flush=True)

    payload = {'rows': rows, 'adapter': adapter, 'prompts': prompts, 'threshold': args.threshold,
               'tile_side': args.tile_side, 'match_iou': args.match_iou, 'runs': args.runs,
               'panel_seed': args.seed, 'vocabulary': args.vocabulary,
               'verifier': verifier.capabilities(),
               'unmapped_spans': getattr(verifier, 'unmapped_spans', {}),
               'wall_seconds': round(time.perf_counter() - started, 1)}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f'{len(rows)} rows -> {args.out} in {payload["wall_seconds"]}s')


if __name__ == '__main__':
    main()
