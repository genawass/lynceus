"""Render reference-vs-prediction overlays for one or more lynceus run sets.

Nonauthoritative visualization. Matching uses the evaluator's one-to-one rule, so the colours mean
exactly what the metrics mean; nothing here is evidence of quality on its own.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.evaluation import match  # noqa: E402
from build_benchmark import references, select_images  # noqa: E402

# Localization and naming are scored separately, so they are coloured separately: a box in the
# right place with the wrong name is a naming failure, not a miss.
GREEN = (70, 225, 100)    # matched a reference and the label is permitted
AMBER = (250, 205, 60)    # matched a reference but the label is not permitted
BLUE = (90, 170, 255)     # matched, but abstained with unknown_object
RED = (240, 65, 65)       # unmatched, inside the declared category scope
ORANGE = (255, 140, 40)   # unmatched, label outside the scope the reference can adjudicate
GRAY, WHITE = (120, 120, 125), (245, 245, 248)
BAR = 46


def load_font(size):
    for candidate in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                      '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT, LABEL_FONT = load_font(17), load_font(13)


def panel(image, title, subtitle, boxes):
    canvas = Image.new('RGB', (image.width, image.height + BAR), (16, 16, 18))
    canvas.paste(image, (0, BAR))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill=WHITE, font=TITLE_FONT)
    draw.text((8, 26), subtitle, fill=(185, 185, 192), font=LABEL_FONT)
    for box, colour in boxes:
        draw.rectangle([box[0], box[1] + BAR, box[2], box[3] + BAR], outline=colour, width=2)
    return canvas


def load_predictions(run_dir, min_score, show):
    """Retained objects, optionally reduced to the ones worth looking at.

    A run keeps every proposal it did not reject, so the full set is dominated by low-score boxes
    that obscure the comparison. Filtering here is a display choice only: it never changes the
    annotation, and the counts printed alongside say how many boxes were hidden.
    """
    result = json.loads((run_dir / 'annotation.json').read_text())
    scores = {}
    candidates = run_dir / 'evidence/candidates.jsonl'
    if candidates.is_file():
        for line in candidates.read_text().splitlines():
            entry = json.loads(line)
            if entry['disposition'] == 'retained':
                scores[tuple(round(v, 3) for v in entry['bbox_xyxy'])] = entry['model_score']
    preds = []
    for o in result['objects']:
        score = scores.get(tuple(round(v, 3) for v in o['bbox_xyxy']))
        preds.append({'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id'],
                      'status': o['status'], 'score': score})
    total = len(preds)
    if show == 'accepted':
        preds = [p for p in preds if p['status'] == 'accepted']
    if min_score is not None:
        preds = [p for p in preds if p['score'] is None or p['score'] >= min_score]
    return result, preds, total


def render(stem, root, run_sets, mapping, out_dir, scale, min_score, show):
    path = root / 'images/val' / f'{stem}.jpg'
    with Image.open(path) as im:
        image = im.convert('RGB')
    width, height = image.size
    refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'], width, height)
    eligible = [r for r in refs if r.get('eligible', True)]
    ignored = [r for r in refs if not r.get('eligible', True)]
    scope = set(mapping['category_scope'])

    panels, row = [], {'image': stem, 'eligible': len(eligible), 'ignored': len(ignored)}
    found_by_any = set()
    rendered = []
    for name, runs in run_sets:
        result, preds, total = load_predictions(runs / stem, min_score, show)
        pairs = match(preds, eligible, .5)
        hit_ref = {b for _, b, _ in pairs}
        found_by_any |= hit_ref
        # matched predictions, split by whether the emitted label is one the reference permits
        named, misnamed, abstained = {}, {}, {}
        for a, b, _ in pairs:
            label = preds[a]['label']
            if label in ('unknown_object', 'entity'):
                abstained[a] = b
            elif label in eligible[b]['permitted_labels']:
                named[a] = b
            else:
                misnamed[a] = b
        if show == 'matched':
            keep = {a for a, _, _ in pairs}
            preds = [p for i, p in enumerate(preds) if i in keep]
            pairs = match(preds, eligible, .5)
            named, misnamed, abstained = {}, {}, {}
            for a, b, _ in pairs:
                label = preds[a]['label']
                if label in ('unknown_object', 'entity'):
                    abstained[a] = b
                elif label in eligible[b]['permitted_labels']:
                    named[a] = b
                else:
                    misnamed[a] = b
        out_of_scope = sum(1 for p in preds if p['label'] not in scope and p['label'] not in ('unknown_object', 'entity'))
        row[name] = {'retained': len(preds), 'hidden': total - len(preds), 'matched': len(pairs), 'named': len(named),
                     'misnamed': len(misnamed), 'abstained': len(abstained),
                     'missed': len(eligible) - len(hit_ref), 'out_of_scope': out_of_scope,
                     'views': result['search']['mandatory_completed'],
                     'wall_seconds': result['summary']['resources']['wall_seconds']}

        def colour(index, label):
            if index in named:
                return GREEN
            if index in misnamed:
                return AMBER
            if index in abstained:
                return BLUE
            return RED if label in scope or label in ('unknown_object', 'entity') else ORANGE
        rendered.append((name, result, preds, hit_ref, row[name],
                         [(p['bbox_xyxy'], colour(i, p['label'])) for i, p in enumerate(preds)]))

    panels.append(panel(image, f'{stem} - VisDrone reference',
                        f'{len(eligible)} eligible  green=found  red=missed by every run  gray=ignored region',
                        [(r['bbox_xyxy'], GREEN if i in found_by_any else RED) for i, r in enumerate(eligible)]
                        + [(r['bbox_xyxy'], GRAY) for r in ignored]))
    for name, result, preds, hit_ref, counts, boxes in rendered:
        ontology = result['ontology']['id']
        panels.append(panel(image, f'{stem} - lynceus {name} ({ontology}, unqualified)',
                            f"{len(preds)} shown ({counts['hidden']} hidden)  {len(hit_ref)}/{len(eligible)} refs hit  |  "
                            f"green named {counts['named']}  amber misnamed {counts['misnamed']}  "
                            f"blue abstained {counts['abstained']}  red unmatched  orange out of scope",
                            boxes))

    gap = 8
    sheet = Image.new('RGB', (sum(p.width for p in panels) + gap * (len(panels) - 1), panels[0].height), (16, 16, 18))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + gap
    if scale != 1.0:
        sheet = sheet.resize((int(sheet.width * scale), int(sheet.height * scale)), Image.LANCZOS)
    sheet.save(out_dir / f'{stem}.png', optimize=True)
    row['file'] = f'{stem}.png'
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping.json'))
    parser.add_argument('--runs', action='append', required=True, metavar='NAME=DIR',
                        help='repeatable; each adds a prediction panel')
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--min-score', type=float, default=None,
                        help='hide retained boxes below this detector score (display only)')
    parser.add_argument('--show', choices=['all', 'accepted', 'matched'], default='all',
                        help="'accepted' keeps only objects the acceptance rule accepted; "
                             "'matched' keeps only those matching a reference")
    args = parser.parse_args()

    run_sets = []
    for spec in args.runs:
        name, _, directory = spec.partition('=')
        run_sets.append((name, Path(directory)))
    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in select_images(root, args.count, args.seed):
        stem = Path(name).stem
        if any(not (runs / stem / 'annotation.json').is_file() for _, runs in run_sets):
            print(f'skip {stem}: missing run')
            continue
        rows.append(render(stem, root, run_sets, mapping, out_dir, args.scale, args.min_score, args.show))
        summary = '  '.join(
            f"{k}[hit {v['matched']}/{rows[-1]['eligible']}, named {v['named']}, misnamed {v['misnamed']}, abstained {v['abstained']}]"
            for k, v in rows[-1].items() if isinstance(v, dict))
        print(f"{stem}  {summary}")
    (out_dir / 'summary.json').write_text(json.dumps(rows, indent=2))
    print(f'\n{len(rows)} comparisons -> {out_dir}')


if __name__ == '__main__':
    main()
