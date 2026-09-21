"""Run the frozen VisDrone panel into one run directory per image.

The image selection comes from the mapping, so this script chooses nothing. Its only job is to
apply one configuration to every image of the panel and leave the evidence on disk, which the
diagnostics then read. A probe run is the same call with a lower threshold: the adapter applies
the score threshold inside `predict`, so observations below it never reach the candidate store and
the only way to see them is to run again lower.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lynceus.pipeline import annotate  # noqa: E402
from build_benchmark import select_images  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', default=str(Path(__file__).parent / 'mapping-strict.json'))
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', required=True, help='directory to hold one run directory per image stem')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1, help='1 is the held-out panel')
    parser.add_argument('--threshold', type=float, default=0.1, help='lower it to probe for sub-threshold proposals')
    parser.add_argument('--tile-levels', type=int, default=1)
    parser.add_argument('--tile-overlap', type=float, default=0.25)
    parser.add_argument('--merge-ios', type=float, default=0.6)
    parser.add_argument('--merge-iou', type=float, default=0.5)
    parser.add_argument('--vocabulary', help='per-model prompt vocabulary id')
    parser.add_argument('--device')
    parser.add_argument('--limit', type=int, help='stop after this many images, for a timing probe')
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    root = Path(mapping['reference_root'])
    stems = [Path(name).stem for name in select_images(root, args.count, args.seed)]
    if args.limit:
        stems = stems[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    for index, stem in enumerate(stems, start=1):
        run_dir = out / stem
        if (run_dir / 'annotation.json').is_file():
            print(f'[{index}/{len(stems)}] {stem} already present, skipped', flush=True)
            continue
        result = annotate(str(root / 'images/val' / f'{stem}.jpg'), args.bundle, str(run_dir),
                          tile_levels=args.tile_levels, tile_overlap=args.tile_overlap,
                          merge_ios=args.merge_ios, merge_iou=args.merge_iou,
                          device=args.device, threshold=args.threshold,
                          ontology_id=mapping['ontology'], vocabulary_id=args.vocabulary)
        print(f'[{index}/{len(stems)}] {stem}: {len(result["objects"])} retained, '
              f'{result["summary"]["resources"]["raw_observations"]} observations, '
              f'{result["summary"]["resources"]["wall_seconds"]:.1f}s', flush=True)
    print(f'panel complete in {time.perf_counter() - started:.1f}s -> {out}')


if __name__ == '__main__':
    main()
