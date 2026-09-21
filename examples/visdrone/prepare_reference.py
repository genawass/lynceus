"""Prepare an exhaustive reference panel for annotation, and ingest it once annotated.

The finite-category ceiling blocks precision, calibration and granularity conformance together.
Only an exhaustively annotated panel lifts it, and the protocol that makes such a panel trustworthy
is mostly procedure: sweep every tile, annotate before seeing predictions, record ambiguity rather
than deciding it, tag granularity, carry leakage identity.

`prepare` lays that procedure out as files a person can work through: one review crop per tile at
source resolution, and a skeleton to fill. `ingest` checks the filled result against every
obligation and emits a benchmark payload.

The default selection is the held-out panel the system has already been measured on, because
annotating those images exhaustively converts the existing precision bound into a number rather
than producing a separate result that cannot be compared to anything.

Nothing here shows the annotator a prediction. That is the point: a reference built by reviewing
predictions measures agreement, and the anchoring does not show up in the finished file.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.artifacts import digest  # noqa: E402
from lynceus.reference import build_panel, plan_review  # noqa: E402
from lynceus.review_ui import render_review_page  # noqa: E402
from build_benchmark import select_images  # noqa: E402


def prepare(root, stems, out, tile, overlap, ontology_id, runs=None):
    from lynceus.policy import available_ontologies
    ontology = available_ontologies()[ontology_id]
    out.mkdir(parents=True, exist_ok=True)
    pages = []
    if runs:
        print(f'verification mode: proposals from {runs}')
        print('this panel will measure precision and cannot measure recall\n')
    for stem in stems:
        path = root / 'images/val' / f'{stem}.jpg'
        with Image.open(path) as handle:
            image = handle.convert('RGB')
            width, height = image.size
            plan = plan_review(width, height, tile, overlap)
            crops = out / stem / 'tiles'
            crops.mkdir(parents=True, exist_ok=True)
            for entry in plan:
                x1, y1, x2, y2 = (int(v) for v in entry['bbox_xyxy'])
                image.crop((x1, y1, x2, y2)).save(crops / f"{entry['id']}.png")
        skeleton = {
            'id': stem,
            'sha256': digest(path.read_bytes()),
            # VisDrone frames of one sequence are near duplicates, so the sequence prefix is the
            # scene identity; the panel carries it so leakage against any other split is checkable.
            'scene_id': stem.split('_')[0],
            'near_duplicate_group': stem.split('_')[0],
            'size': [width, height],
            'domain': 'aerial_drone',
            'split': 'evaluation',
            'ontology': ontology_id,
            'visited_tiles': [],
            'annotations': [],
            'how_to_fill': {
                'visited_tiles': 'add each tile id once it has been reviewed; every planned tile must appear',
                'annotations': 'one entry per entity, in whole-image pixel coordinates',
                'entry': {'id': 'unique within this image', 'bbox_xyxy': '[x_min, y_min, x_max, y_max]',
                          'label': f'an id from the {ontology_id} ontology',
                          'kind': 'instance | part | group | stuff',
                          'scene_layer': 'physical | reflected | depicted',
                          'resolution': 'resolved, or unresolved where the policy does not settle extent or granularity',
                          'permitted_labels': 'optional; broader labels an annotator may also emit for this entity'},
                'do_not': 'open the system output before this file is finished',
            },
            'planned_tiles': [entry['id'] for entry in plan],
        }
        (out / stem / 'reference.json').write_text(json.dumps(skeleton, indent=1))
        # The page sits beside a copy of the image so it opens over file:// with no server and no
        # network, which is the only way it runs under the same offline conditions as the annotator.
        with Image.open(path) as handle:
            handle.convert('RGB').save(out / stem / 'image.jpg', quality=95)
        predictions = None
        if runs:
            result = json.loads((Path(runs) / stem / 'annotation.json').read_text())
            # Geometry and class only. A score beside a box would steer the annotator toward the
            # model's own confidence, which is the judgement being tested.
            predictions = [{'id': o['id'], 'bbox_xyxy': o['bbox_xyxy'], 'label': o['label']['id']}
                           for o in result['objects']
                           if o['kind'] == 'instance' and o['scene_layer'] == 'physical']
            skeleton['anchoring'] = 'prediction_assisted'
            skeleton['proposals'] = len(predictions)
            (out / stem / 'reference.json').write_text(json.dumps(skeleton, indent=1))
        (out / stem / 'review.html').write_bytes(
            render_review_page(skeleton, plan, ontology, 'image.jpg', predictions))
        pages.append((stem, len(plan), len(predictions) if predictions is not None else None))
        print(f'  {stem}: {len(plan)} tiles' +
              (f', {len(predictions)} proposals to judge' if predictions is not None else '') +
              f' -> {out / stem}')

    rows = ''.join(f'<li><a href="{s}/review.html">{s}</a> <small>{n} tiles'
                   + (f', {k} proposals' if k is not None else '') + '</small></li>' for s, n, k in pages)
    caution = ('<p style="border-left:3px solid #e8b339;padding-left:10px"><b>Verification panel.</b> '
               'Every box you keep is one the system proposed, so this panel measures precision and '
               'cannot measure recall. Add anything you notice that is missing — it is kept and counted — '
               'but noticing is not a sweep, so the result stays anchored.</p>') if pages and pages[0][2] is not None else ''
    (out / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Reference panel</title>'
        '<style>body{font:14px/1.6 system-ui;background:#16161a;color:#e8e8ee;margin:40px auto;max-width:640px}'
        'a{color:#5aa9ff}small{color:#9a9aa8}li{margin:4px 0}</style>'
        f'<h1>Exhaustive reference panel</h1><p>{len(pages)} images. Open each, sweep every tile, save '
        '<code>reference.json</code> back into that image\'s folder, then run <code>prepare_reference.py ingest</code>.</p>'
        + (caution if caution else
           '<p><b>Do not open the system\'s output until the panel is finished.</b> A reference annotated '
           'against predictions measures agreement, and the anchoring does not show up in the result.</p>')
        + f'<ul>{rows}</ul>')
    print(f'\nprepared {len(stems)} images under {out}')
    print(f'open {out / "index.html"} in a browser, annotate, and save each reference.json into its folder')


def ingest(out, ontology, domain, annotator, tile, overlap, destination):
    records = []
    anchorings = set()
    rejected = 0
    for folder in sorted(p for p in out.iterdir() if p.is_dir()):
        path = folder / 'reference.json'
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        record.pop('how_to_fill', None)
        record.pop('planned_tiles', None)
        record.pop('ontology', None)
        record.pop('proposals', None)
        anchorings.add(record.pop('anchoring', 'blind'))
        rejected += len(record.pop('rejected', []) or [])
        records.append(record)
    if not records:
        raise SystemExit(f'no reference.json found under {out}')
    if len(anchorings) > 1:
        raise SystemExit(f'panel mixes anchoring modes {sorted(anchorings)}; build them separately')
    anchoring = anchorings.pop()
    panel = build_panel(records, ontology, domain=domain, annotator=annotator, tile=tile, overlap=overlap,
                        blind=anchoring == 'blind', anchoring=anchoring, rejected=rejected)
    Path(destination).write_text(json.dumps(panel, indent=1))
    counts = panel['counts']
    print(f"ingested {counts['images']} images, {counts['references']} references, "
          f"{counts['eligible_references']} eligible")
    print(f'wrote {destination}')
    if panel['recall_interpretable']:
        print('precision on this panel is a measurement rather than a lower bound: no category scope applies')
    else:
        print(f"anchoring: {panel['anchoring']}, {panel['rejected_predictions']} proposals rejected")
        print('precision is a measurement; recall from this panel is not interpretable and the evaluator says so')
        print(f"origins: {panel['origins']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'serve', 'ingest'])
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict.json'))
    parser.add_argument('--out', required=True, help='working directory for the panel')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1, help='1 is the held-out panel already measured')
    parser.add_argument('--tile', type=int, default=512)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--annotator', default='unnamed')
    parser.add_argument('--benchmark', default='exhaustive-panel.json')
    parser.add_argument('--host', default='127.0.0.1', help='loopback by default; the panel holds imagery')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--with-predictions', dest='runs',
                        help='run directory tree; switches to verification mode, which measures '
                             'precision and cannot measure recall')
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    out = Path(args.out)
    if args.command == 'prepare':
        stems = [Path(name).stem for name in select_images(root, args.count, args.seed)]
        prepare(root, stems, out, args.tile, args.overlap, mapping['ontology'], args.runs)
    elif args.command == 'serve':
        from lynceus.review_server import serve
        server = serve(out, args.host, args.port)
        print(f'serving {out.resolve()} on http://{args.host}:{args.port}')
        print(f'over ssh:  ssh -N -L {args.port}:127.0.0.1:{args.port} <this-host>')
        print(f'then open: http://127.0.0.1:{args.port}/index.html')
        print('saves are written back into the panel; ctrl-c to stop')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nstopped')
    else:
        ingest(out, mapping['ontology'], mapping['domain'], args.annotator, args.tile, args.overlap, args.benchmark)


if __name__ == '__main__':
    main()
