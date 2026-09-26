"""Convert RF100-VL datasets into the VisDrone layout the panel tools read.

Each dataset becomes <out>/<name>/images/val/*.jpg and annotations/val/*.txt with one
`x,y,w,h,1,category,0,0` row per box, so run_panel, verify_panel, fit_verify_threshold and
render_consensus run against it through a mapping file alone. Image bytes are copied as stored,
never re-encoded. Only the split asked for is read, and RF100-VL's split is recorded as `val` here
only because that is the directory the tools look in.
"""
import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='directory holding data/<split>-*.parquet')
    parser.add_argument('--split', default='test')
    parser.add_argument('--dataset', action='append', required=True, help='RF100-VL dataset_name; repeatable')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    wanted = set(args.dataset)
    counts = {name: {'images': 0, 'boxes': 0, 'categories': {}} for name in wanted}
    for shard in sorted(Path(args.source).glob(f'data/{args.split}-*.parquet')):
        table = pq.read_table(shard, columns=['dataset_name'])
        names = table.column('dataset_name').to_pylist()
        rows = [i for i, n in enumerate(names) if n in wanted]
        if not rows:
            continue
        table = pq.read_table(shard).take(rows).to_pylist()
        for row in table:
            name = row['dataset_name']
            root = Path(args.out) / name
            (root / 'images/val').mkdir(parents=True, exist_ok=True)
            (root / 'annotations/val').mkdir(parents=True, exist_ok=True)
            stem = f"{row['dataset_id']}_{Path(row['file_name']).stem}"[:120]
            data = row['image']['bytes']
            with Image.open(io.BytesIO(data)) as image:
                fmt, size = image.format, image.size
            if fmt == 'JPEG':
                (root / 'images/val' / f'{stem}.jpg').write_bytes(data)
            else:  # the tools glob *.jpg; a lossless source is re-encoded at the highest quality
                with Image.open(io.BytesIO(data)) as image:
                    image.convert('RGB').save(root / 'images/val' / f'{stem}.jpg', quality=100, subsampling=0)
            if size != (row['width'], row['height']):
                raise ValueError(f'{stem}: stored size {size} differs from recorded {row["width"]}x{row["height"]}')
            lines = []
            # Stored column-wise: one list per field, not one record per box.
            boxes = row['annotations'] or {'bbox': [], 'category_id': []}
            for (x, y, w, h), category in zip(boxes['bbox'], boxes['category_id']):
                lines.append(f"{round(x)},{round(y)},{round(w)},{round(h)},1,{category},0,0")
                counts[name]['categories'][category] = counts[name]['categories'].get(category, 0) + 1
            (root / 'annotations/val' / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
            counts[name]['images'] += 1
            counts[name]['boxes'] += len(lines)
    for name, c in sorted(counts.items()):
        print(f"{name}: {c['images']} images, {c['boxes']} boxes, by category id {dict(sorted(c['categories'].items()))}")
    (Path(args.out) / f'import-{args.split}.json').write_text(json.dumps(
        {'source': str(Path(args.source).resolve()), 'split': args.split, 'counts': counts}, indent=2))


if __name__ == '__main__':
    main()
