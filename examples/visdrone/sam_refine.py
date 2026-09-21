"""Refine retained boxes against pixels with a box-prompted SAM, and measure what it returns.

Miss attribution put 55 of 71 misses in `localized_below_match`, and boundary headroom then showed
that no selection among recorded observations recovers them with any signal the pipeline has. What
is left is the requirement the annotation policy states directly: boundaries refined against
pixels. This script tests exactly that, and nothing else.

Each retained box is used as a box prompt. The mask SAM returns under its own best-quality
estimate is converted back to a tight box, and the refined boxes are written out for the same
diagnostics the unrefined ones went through. No box is added and none is removed, so recall can
only move through geometry -- which is the claim under test.

Two backends are supported and produce directly comparable output. SAM 1 is ungated and runs on
the pinned `transformers`. SAM 3 is gated and needs a newer `transformers`, supplied on
`PYTHONPATH` so the pinned stack and its bundle manifest stay untouched; its `Sam3TrackerModel` is
the box-prompt path, since `Sam3Model` with `input_boxes` returns DETR-style detections rather
than a mask per prompt.

DART accelerates SAM 3 by sharing the class-agnostic backbone across class prompts. That applies
to multi-class detection, not to this: refinement issues one prompt per box with no class loop, so
there is no backbone cost to amortise across classes. Being training-free, it would change the
cost of this measurement and not its result.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_benchmark import select_images  # noqa: E402


def mask_to_box(mask):
    """Tight box around the mask, or None when the mask is empty.

    The whole mask is used rather than its largest component: the annotation policy keeps
    disconnected visible portions of one object inside its box, so discarding a component here
    would encode a policy this refinement step is not entitled to decide.
    """
    rows = np.any(mask, axis=1)
    columns = np.any(mask, axis=0)
    if not rows.any() or not columns.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(columns)[0][[0, -1]]
    return [float(x1), float(y1), float(x2 + 1), float(y2 + 1)]


def load_backend(backend, path, device):
    """The model, its processor, and how to read a mask out of its output."""
    if backend == 'sam3':
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        model = Sam3TrackerModel.from_pretrained(path, local_files_only=True).eval().to(device)
        return model, Sam3TrackerProcessor.from_pretrained(path, local_files_only=True)
    from transformers import SamModel, SamProcessor
    model = SamModel.from_pretrained(path, local_files_only=True).eval().to(device)
    return model, SamProcessor.from_pretrained(path, local_files_only=True)


def decode(output, inputs, processor, backend):
    """Masks and per-prompt quality scores, in the view's own coordinates."""
    if backend == 'sam3':
        masks = processor.post_process_masks(output.pred_masks.cpu(), inputs['original_sizes'].cpu())[0]
    else:
        masks = processor.image_processor.post_process_masks(
            output.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())[0]
    return masks, output.iou_scores.cpu()[0]


def forward(model, inputs, amp):
    """One SAM forward, optionally under autocast.

    Half-precision activations are what make this affordable: the attention maps, not the weights,
    are what exhaust a 12GB card at 1024 square, so autocast restores batching without changing
    the stored weights. It does change numerics, so both refinement modes are run the same way and
    a result from one precision is never compared against a result from the other.
    """
    with torch.no_grad():
        if amp:
            with torch.autocast('cuda', dtype=torch.float16):
                return model(**inputs, multimask_output=True)
        return model(**inputs, multimask_output=True)


def best_box(output, inputs, processor, index_in_batch):
    """The mask SAM rates highest for one prompt, as a tight box in that view's coordinates."""
    masks = processor.image_processor.post_process_masks(
        output.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())
    scores = output.iou_scores.cpu()
    # SAM proposes several masks per prompt and scores its own quality; take the one it rates
    # highest rather than the largest, which would bias toward over-segmentation.
    best = int(torch.argmax(scores[index_in_batch][0]))
    return mask_to_box(masks[index_in_batch][0][best].numpy())


def refine_whole_image(model, processor, image, boxes, device, amp, backend, batch=32):
    """Prompt every box against one pass over the whole image.

    Cheap, and wrong for this panel: the processor resizes the view to 1024 on its longest side,
    so a twelve-pixel object in a two-thousand-pixel image reaches the model at about six pixels.
    Kept so the comparison against crop refinement is a measurement rather than an assertion.
    """
    refined = []
    for start in range(0, len(boxes), batch):
        chunk = [[float(v) for v in box] for box in boxes[start:start + batch]]
        inputs = processor(image, input_boxes=[chunk], return_tensors='pt').to(device)
        output = forward(model, inputs, amp)
        masks, scores = decode(output, inputs, processor, backend)
        for index in range(len(chunk)):
            best = int(torch.argmax(scores[index]))
            refined.append(mask_to_box(masks[index][best].numpy()))
    return refined


def plan_windows(width, height, side, overlap=0.5):
    """Overlapping source-resolution windows covering the image, like the scan planner's tiles."""
    step = max(1, int(side * (1 - overlap)))
    xs = sorted({min(x, max(0, width - side)) for x in range(0, max(1, width), step)})
    ys = sorted({min(y, max(0, height - side)) for y in range(0, max(1, height), step)})
    return [[x, y, min(x + side, width), min(y + side, height)] for y in ys for x in xs]


def assign(box, windows):
    """The window that contains the box with the most room to spare, or None if none contains it.

    Margin decides rather than mere containment, so an object sits as far from a view border as the
    schedule allows. A box touching the edge of its view is the truncation case the merge rule
    already has to work around, and refinement should not manufacture more of it.
    """
    best, best_margin = None, -1.0
    for index, window in enumerate(windows):
        if box[0] >= window[0] and box[1] >= window[1] and box[2] <= window[2] and box[3] <= window[3]:
            margin = min(box[0] - window[0], box[1] - window[1], window[2] - box[2], window[3] - box[3])
            if margin > best_margin:
                best, best_margin = index, margin
    return best


def refine_on_tiles(model, processor, image, boxes, device, side, amp, backend, max_prompts=64):
    """Refine boxes inside overlapping source-resolution tiles, then map back.

    One encoder pass serves every box in a tile, which is what makes source-resolution refinement
    affordable: a per-box crop pays a full 1024-square encoder pass to upscale a 128-pixel window,
    buying nothing the tile did not already carry. A tile also gives the object real magnification
    -- a 512-pixel view upscales to 1024, where the whole frame is downscaled to it.

    A box no tile fully contains keeps its original geometry rather than being refined against a
    view that truncates it.
    """
    width, height = image.size
    windows = plan_windows(width, height, side)
    refined = [None] * len(boxes)
    groups = {}
    for index, box in enumerate(boxes):
        window = assign(box, windows)
        if window is not None:
            groups.setdefault(window, []).append(index)
    for window_index, indices in groups.items():
        x1, y1, x2, y2 = windows[window_index]
        crop = image.crop((x1, y1, x2, y2))
        for start in range(0, len(indices), max_prompts):
            group = indices[start:start + max_prompts]
            prompts = [[[float(boxes[i][0] - x1), float(boxes[i][1] - y1),
                         float(boxes[i][2] - x1), float(boxes[i][3] - y1)] for i in group]]
            inputs = processor(crop, input_boxes=prompts, return_tensors='pt').to(device)
            output = forward(model, inputs, amp)
            masks, scores = decode(output, inputs, processor, backend)
            for position, index in enumerate(group):
                best = int(torch.argmax(scores[position]))
                local = mask_to_box(masks[position][best].numpy())
                if local is not None:
                    # The model's own quality estimate travels with the box: it is evidence
                    # available at inference, so a rule may gate on it without seeing a reference.
                    refined[index] = {'box': [local[0] + x1, local[1] + y1, local[2] + x1, local[3] + y1],
                                      'quality': float(scores[position][best])}
    return refined


def refine(model, processor, image, boxes, device, tile_side, amp, backend='sam1'):
    if tile_side is None:
        return refine_whole_image(model, processor, image, boxes, device, amp, backend)
    return refine_on_tiles(model, processor, image, boxes, device, tile_side, amp, backend)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict.json'))
    parser.add_argument('--runs', required=True, help='shipped run directory tree')
    parser.add_argument('--model', default='/home/genadiy/data/models/sam-vit-huge')
    parser.add_argument('--backend', choices=['sam1', 'sam3'], default='sam1')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tile-side', type=int, default=512,
                        help='source-resolution tile side; a 512 view upscales to SAM\'s 1024 input, '
                             'where the whole frame would be downscaled to it')
    parser.add_argument('--whole-image', action='store_true',
                        help='refine against the full view instead of per-box crops')
    parser.add_argument('--fp32', action='store_true', help='disable autocast; slower and needs a smaller batch')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    stems = [Path(name).stem for name in select_images(root, args.count, args.seed)]

    model, processor = load_backend(args.backend, args.model, args.device)

    out, empty, total = {}, 0, 0
    for stem in stems:
        records = [json.loads(line) for line
                   in (Path(args.runs) / stem / 'evidence/candidates.jsonl').read_text().splitlines() if line]
        retained = [r for r in records if r['disposition'] == 'retained']
        with Image.open(root / 'images/val' / f'{stem}.jpg') as handle:
            image = handle.convert('RGB')
            boxes = refine(model, processor, image, [r['bbox_xyxy'] for r in retained], args.device,
                           None if args.whole_image else args.tile_side, not args.fp32, args.backend)
        for record, box in zip(retained, boxes):
            out[f'{stem}/{record["id"]}'] = box
            if box is None:
                empty += 1
        total += len(retained)
        print(f'  {stem}: {len(retained)} boxes refined', flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f'\nrefined {total - empty} of {total}; {empty} masks were empty and keep their original box')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
