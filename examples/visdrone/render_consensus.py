"""Draw what each agreement rule keeps, side by side with the reference.

Consensus is a filter over the same retained set, so the only thing that changes between panels is
which boxes survive. Putting them beside each other shows what each rule costs in a way a pair of
numbers does not: whether the boxes a rule discards are the ones a reader would also discard.

Colour separates the two populations the reference can and cannot adjudicate. Green is a box
matching an eligible reference, which the rule was right to keep. Red is a box matching none,
which is either a false positive or a real object VisDrone never annotated -- the ambiguity that
makes precision a bound here, and it is left visible rather than resolved by colour.
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

GREEN, RED, GRAY, WHITE = (70, 225, 100), (240, 65, 65), (120, 120, 125), (245, 245, 248)
BAR = 46


def load_font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
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


def confirmations(paths):
    """One lookup per verifier: (image, object id) -> confirmed."""
    table = {}
    for name, path in paths:
        rows = json.loads(Path(path).read_text())['rows']
        table[name] = {(r['image'], r['id']): r['confirmed'] for r in rows}
    return table


RULES = {
    'all retained': lambda votes: True,
    'either confirms': lambda votes: any(votes),
    'agree (all confirm)': lambda votes: all(votes),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict-v2.json'))
    parser.add_argument('--runs', required=True)
    parser.add_argument('--verifier', action='append', required=True, metavar='NAME=ROWS.json',
                        help='repeatable; each adds a verifier whose confirmations form the rules')
    parser.add_argument('--external', action='append', default=[], metavar='NAME=DIR',
                        help='repeatable; DIR/<stem>.json is a list of {"label", "box": [x1, y1, x2, y2]} '
                             'from a detector outside the pipeline, drawn as its own panel')
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--columns', type=int, help='panels per row; default 3')
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    paths = [(spec.split('=', 1)[0], spec.split('=', 1)[1]) for spec in args.verifier]
    table = confirmations(paths)
    names = [name for name, _ in paths]
    rules = dict(RULES)
    for name in names:
        rules[f'{name} only'] = (lambda n: (lambda votes, n=n: votes[names.index(n)]))(name)
    order = ['all retained'] + [f'{n} only' for n in names] + (
        ['either confirms', 'agree (all confirm)'] if len(names) > 1 else [])

    externals = [(spec.split('=', 1)[0], Path(spec.split('=', 1)[1])) for spec in args.external]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in select_images(root, args.count, args.seed):
        stem = Path(name).stem
        result = json.loads((Path(args.runs) / stem / 'annotation.json').read_text())
        with Image.open(root / 'images/val' / f'{stem}.jpg') as handle:
            image = handle.convert('RGB')
        refs, _ = references(root / 'annotations/val' / f'{stem}.txt', mapping['categories'],
                             *result['image']['normalized_size'])
        eligible = [r for r in refs if r.get('eligible', True)]
        objects = [{'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id'], 'id': o['id']}
                   for o in result['objects']]
        votes = {o['id']: [table[n].get((stem, o['id']), False) for n in names] for o in objects}

        panels, row = [], {'image': stem, 'eligible': len(eligible)}
        found_by_any = set()
        rendered = []
        for rule in order:
            kept = [o for o in objects if rules[rule](votes[o['id']])]
            pairs = match(kept, eligible, .5)
            hit_pred = {a for a, _, _ in pairs}
            found_by_any |= {b for _, b, _ in pairs}
            row[rule] = {'kept': len(kept), 'matched': len(pairs),
                         'recall': round(len(pairs) / len(eligible), 4) if eligible else None,
                         'precision': round(len(pairs) / len(kept), 4) if kept else None}
            rendered.append((rule, kept, row[rule],
                             [(o['bbox_xyxy'], GREEN if i in hit_pred else RED)
                              for i, o in enumerate(kept)]))
        # An external detector is not a rule over the retained set, so it shares the reference and
        # the colouring but not the votes; an image it did not cover simply gets no panel.
        for ext, directory in externals:
            path = directory / f'{stem}.json'
            if not path.exists():
                continue
            kept = [{'bbox_xyxy': d['box'], 'label': d['label']} for d in json.loads(path.read_text())]
            pairs = match(kept, eligible, .5)
            hit_pred = {a for a, _, _ in pairs}
            found_by_any |= {b for _, b, _ in pairs}
            row[ext] = {'kept': len(kept), 'matched': len(pairs),
                        'recall': round(len(pairs) / len(eligible), 4) if eligible else None,
                        'precision': round(len(pairs) / len(kept), 4) if kept else None}
            rendered.append((ext, kept, row[ext],
                             [(o['bbox_xyxy'], GREEN if i in hit_pred else RED)
                              for i, o in enumerate(kept)]))

        panels.append(panel(image, f'{stem} - VisDrone reference',
                            f'{len(eligible)} eligible  green=found by some rule  red=found by none',
                            [(r['bbox_xyxy'], GREEN if i in found_by_any else RED)
                             for i, r in enumerate(eligible)]))
        for rule, kept, counts, boxes in rendered:
            panels.append(panel(image, f'{stem} - {rule}',
                                f'{counts["kept"]} kept  {counts["matched"]}/{len(eligible)} refs hit  '
                                f'recall {counts["recall"]}  precision {counts["precision"]}  '
                                'green=matches a reference  red=unmatched (false, or real and unannotated)',
                                boxes))

        # A row of seven panels is unreadable at any sane width, so lay them out in a grid.
        gap = 8
        columns = args.columns or min(len(panels), 3)
        rows_needed = (len(panels) + columns - 1) // columns
        cell_w, cell_h = panels[0].width, panels[0].height
        sheet = Image.new('RGB', (columns * cell_w + gap * (columns - 1),
                                  rows_needed * cell_h + gap * (rows_needed - 1)), (16, 16, 18))
        for index, p in enumerate(panels):
            column, line = index % columns, index // columns
            sheet.paste(p, (column * (cell_w + gap), line * (cell_h + gap)))
        if args.scale != 1.0:
            sheet = sheet.resize((int(sheet.width * args.scale), int(sheet.height * args.scale)), Image.LANCZOS)
        sheet.save(out / f'{stem}.png', optimize=True)
        summary.append(row)
        print(stem + '  ' + '  '.join(
            f'{k}[{v["kept"]}k {v["matched"]}m]' for k, v in row.items() if isinstance(v, dict)))

    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\n{len(summary)} sheets -> {out}')


if __name__ == '__main__':
    main()
