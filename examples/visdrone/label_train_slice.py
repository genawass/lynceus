"""Auto-label the declared train slice: one run directory per image, models loaded once.

This is the labelling half of the student experiment in `student-protocol.json`. It chooses nothing:
the slice, the configuration and the leakage rule come from that file, which was frozen before any
label existed. Stricter verifier thresholds are not separate runs -- the verifier's score travels in
each candidate record, so one pass serves every threshold arm and the arms differ by a filter rather
than by an inference.

Sequence-level exclusion against val is the part worth stating twice. Frames of one VisDrone sequence
are near duplicates, so training on a sequence that also appears in val would raise both arms by a
similar amount and still make the comparison a measurement of leakage.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.pipeline import annotate  # noqa: E402
from run_panel import load_models  # noqa: E402


def sequence(stem):
    return stem.split('_')[0]


def select_slice(root, count, seed):
    """The declared selection, recomputed rather than stored, so the rule is the record."""
    import random
    held = {sequence(Path(line).stem) for line in (root / 'val.txt').read_text().split() if line}
    pool = sorted(stem for stem in {Path(line).stem for line in (root / 'train.txt').read_text().split() if line}
                  if sequence(stem) not in held)
    if len(pool) < count:
        raise SystemExit(f'pool has {len(pool)} images, {count} requested')
    return sorted(random.Random(seed).sample(pool, count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default=str(Path(__file__).parent / 'student-protocol.json'))
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', required=True, help='directory to hold one run directory per image stem')
    parser.add_argument('--refine-bundle')
    parser.add_argument('--verify-bundle')
    parser.add_argument('--device')
    parser.add_argument('--limit', type=int, help='stop after this many images, for a timing probe')
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    configuration = protocol['labelling_configuration']
    root = Path(protocol['reference_root'])
    stems = select_slice(root, protocol['train_slice']['count'], protocol['train_slice']['seed'])
    if args.limit:
        stems = stems[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'slice.json').write_text(json.dumps(
        {'protocol': protocol['id'], 'stems': stems,
         'sequences': sorted({sequence(s) for s in stems})}, indent=1))

    # `load_models` reads these off one namespace, so the protocol's configuration is passed as one.
    args.threshold = configuration['threshold']
    args.vocabulary = None
    args.refine_tile_side = None
    args.verify_threshold = configuration['verify_threshold']
    args.verify_prompt_batch = configuration['verify_prompt_batch']
    adapter, refiner, verifier = load_models(args, protocol['ontology'])

    started = time.perf_counter()
    for index, stem in enumerate(stems, start=1):
        run_dir = out / stem
        if (run_dir / 'annotation.json').is_file():
            print(f'[{index}/{len(stems)}] {stem} already present, skipped', flush=True)
            continue
        result = annotate(str(root / 'images/train' / f'{stem}.jpg'), args.bundle, str(run_dir),
                          adapter=adapter, refiner=refiner, verifier=verifier,
                          tile_levels=configuration['tile_levels'],
                          threshold=configuration['threshold'],
                          device=args.device, ontology_id=protocol['ontology'],
                          refine_bundle=args.refine_bundle, verify_bundle=args.verify_bundle,
                          verify_threshold=configuration['verify_threshold'],
                          verify_prompt_batch=configuration['verify_prompt_batch'])
        resources = result['summary']['resources']
        elapsed = time.perf_counter() - started
        print(f'[{index}/{len(stems)}] {stem}: {len(result["objects"])} retained, '
              f'{resources["wall_seconds"]:.1f}s, '
              f'{elapsed / index:.1f}s/image, '
              f'{(len(stems) - index) * elapsed / index / 60:.0f} min left', flush=True)
    print(f'slice complete in {time.perf_counter() - started:.1f}s -> {out}')


if __name__ == '__main__':
    main()
