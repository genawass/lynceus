"""Verify retained boxes against an independent model, with no human in the loop.

A finite-category reference leaves unmatched boxes undecided: each is either a false positive or a
real object VisDrone never annotated. A second model cannot settle that, because agreement is not
truth -- that is why the contract treats an automatic reference as a pseudo-reference and never as
ground truth.

What makes this more than a vote is that half the population is already labelled. A box matching an
eligible reference is a known true positive, so the verifier's confirmation rate on matched boxes
measures its sensitivity directly, on this data, at this object scale. Applying that to the
unmatched boxes turns a bare agreement count into a bounded statement.

The bound is one-sided and the asymmetry matters. Sensitivity is measurable because known positives
exist; specificity is not, because no box here is a known negative -- that is the whole problem.
So a low confirmation rate among unmatched boxes is consistent both with those boxes being false
and with the verifier failing on them, and only the first is a finding. The report says which
quantities are measured and which are assumed.

SAM 3 is a reasonable verifier here: a different family from the prompted detector under test, with
a presence head built for exactly the question of whether a concept is in the view. Its prompt
vocabulary is configuration under D40 and is recorded with the result.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.audit import unmatched_boxes  # noqa: E402
from lynceus.diagnostics import size_bin  # noqa: E402
from lynceus.evaluation import iou_matrix, match_matrix  # noqa: E402
from lynceus.geometry import iou  # noqa: E402
from lynceus.policy import available_ontologies  # noqa: E402
from lynceus.vocabulary import check_adapter, load_vocabulary, resolve_prompts  # noqa: E402
from build_benchmark import references, select_images  # noqa: E402


def plan_tiles(width, height, side, overlap=0.5):
    step = max(1, int(side * (1 - overlap)))
    xs = sorted({min(x, max(0, width - side)) for x in range(0, max(1, width), step)})
    ys = sorted({min(y, max(0, height - side)) for y in range(0, max(1, height), step)})
    return [(x, y, min(x + side, width), min(y + side, height)) for y in ys for x in xs]


def detect(model, processor, image, prompts, labels, device, side, threshold):
    """Every SAM 3 detection over the image, tile by tile, mapped back to source coordinates.

    Tiling for the same reason refinement tiles: a whole-frame pass delivers a twelve-pixel object
    at a fraction of its pixels, and a verifier that cannot see the object is not evidence about it.
    """
    found = []
    for x1, y1, x2, y2 in plan_tiles(*image.size, side):
        view = image.crop((x1, y1, x2, y2))
        for prompt in prompts:
            inputs = processor(images=view, text=prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    output = model(**inputs)
            result = processor.post_process_instance_segmentation(
                output, threshold=threshold, target_sizes=[view.size[::-1]])[0]
            for box, score in zip(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist()):
                found.append({'bbox_xyxy': [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1],
                              # The class id, not the phrasing: a comparison must not depend on how
                              # the question happened to be worded.
                              'label': labels[prompt], 'score': float(score)})
    return found


def confirmations(boxes, detections, iou_threshold):
    """For each box, the best-overlapping verifier detection, or None."""
    out = []
    for box in boxes:
        best, best_iou = None, 0.0
        for d in detections:
            v = iou(box['bbox_xyxy'], d['bbox_xyxy'])
            if v >= iou_threshold and v > best_iou:
                best, best_iou = d, v
        out.append(best)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict.json'))
    parser.add_argument('--runs', required=True)
    parser.add_argument('--model', default='.chkpts/sam3')
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--tile-side', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.3, help='verifier detection threshold')
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--vocabulary', help='prompt vocabulary id for the verifier')
    args = parser.parse_args()

    from transformers import Sam3Model, Sam3Processor

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    ontology = available_ontologies()[mapping['ontology']]
    # The verifier gets its own vocabulary, not the detector's: phrasing does not transfer between
    # a CLIP-style text tower and a presence head, and asking both with one string would test
    # whether they respond alike rather than whether they agree about the world.
    resolution = check_adapter(resolve_prompts(ontology, load_vocabulary(args.vocabulary)), 'sam3')
    prompts = list(resolution['prompts'])
    classes = list(resolution['classes'])

    processor = Sam3Processor.from_pretrained(args.model, local_files_only=True)
    model = Sam3Model.from_pretrained(args.model, local_files_only=True).eval().to(args.device)

    rows = []
    for stem in [Path(n).stem for n in select_images(root, args.count, args.seed)]:
        result = json.loads((Path(args.runs) / stem / 'annotation.json').read_text())
        objects = [o for o in result['objects']
                   if o['kind'] == 'instance' and o['scene_layer'] == 'physical']
        retained = [{'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id']} for o in objects]
        with Image.open(root / 'images/val' / f'{stem}.jpg') as handle:
            image = handle.convert('RGB')
            refs, _ = references(root / 'annotations/val' / f'{stem}.txt',
                                 mapping['categories'], *image.size)
            eligible = [r for r in refs if r.get('eligible', True)]
            detections = detect(model, processor, image, prompts, dict(zip(prompts, classes)),
                                args.device, args.tile_side, args.threshold)
        unmatched, _ = unmatched_boxes(retained, eligible, args.match_iou)
        unmatched = set(unmatched)
        confirmed = confirmations(retained, detections, args.match_iou)
        for index, (box, hit) in enumerate(zip(retained, confirmed)):
            rows.append({'image': stem, 'id': objects[index]['id'], 'label': box['label'],
                         'size_bin': size_bin(box['bbox_xyxy']),
                         # Known from the reference: a matched box is a true positive.
                         'known': 'true_positive' if index not in unmatched else 'undecided',
                         'confirmed': hit is not None,
                         'verifier_label': hit['label'] if hit else None,
                         'verifier_score': round(hit['score'], 4) if hit else None})
        print(f'  {stem}: {len(retained)} retained, {len(detections)} verifier detections, '
              f'{sum(1 for r in rows[-len(retained):] if r["confirmed"])} confirmed', flush=True)

    out = Path(args.out)
    out.write_text(json.dumps({'rows': rows, 'prompts': prompts, 'classes': classes,
                               'vocabulary': resolution['vocabulary'],
                               'prompt_fallbacks': resolution['fallbacks'], 'threshold': args.threshold,
                               'tile_side': args.tile_side, 'match_iou': args.match_iou,
                               'verifier': 'sam3', 'runs': str(args.runs), 'panel_seed': args.seed},
                              indent=1))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
