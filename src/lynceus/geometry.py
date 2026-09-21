import math
import numpy as np


def validate_box(box, width, height):
    if len(box) != 4 or not all(math.isfinite(x) for x in box):
        raise ValueError('invalid_box: coordinates must be finite xyxy')
    x1,y1,x2,y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('invalid_box: inverted, empty, or outside source')
    return list(map(float, box))


def transform_box(box, matrix, width, height):
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3,3) or not np.isfinite(m).all() or abs(np.linalg.det(m)) < 1e-12:
        raise ValueError('invalid_transform')
    x1,y1,x2,y2 = box
    corners = m @ np.array([[x1,x2,x1,x2],[y1,y1,y2,y2],[1,1,1,1]])
    if np.any(abs(corners[2]) < 1e-12): raise ValueError('invalid_transform')
    p = corners[:2]/corners[2]
    return validate_box([p[0].min(),p[1].min(),p[0].max(),p[1].max()],width,height)


def iou(a,b):
    inter = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.0


def _origins(total, size, count):
    """Tile origins covering `total` exactly: first at 0, last flush with the far edge."""
    if count <= 1 or size >= total:
        return [0]
    step = (total - size) / (count - 1)
    return sorted({int(round(i * step)) for i in range(count)})


def _round_half_up(value):
    return math.floor(value + 0.5)


def grid_shape(width, height, level):
    """Column and row counts for `level` that keep tiles as square as the image allows.

    A (level+1)x(level+1) grid inherits the image's aspect ratio, and OWLv2 pads every view to a
    square before resizing, so on a 16:9 image such a grid spends ~44% of each forward pass on
    padding. Distributing the same tile budget by aspect instead keeps tiles near square: a 16:9
    image gets 3x2 at level 1, where a square grid would have used 3x3 for the same magnification.
    Square images are unaffected -- they still get 2x2, 3x3, and so on.
    """
    budget = (level + 1) ** 2
    aspect = width / height
    return (max(1, _round_half_up(math.sqrt(budget * aspect))),
            max(1, _round_half_up(math.sqrt(budget / aspect))))


def _tile_span(total, count, overlap):
    """Tile length whose `count` evenly spaced copies cover `total` with `overlap` linear overlap.

    Solving total = span + (count - 1) * span * (1 - overlap) for span, so `overlap` is the actual
    shared fraction of adjacent tiles rather than an enlargement factor.
    """
    if count <= 1:
        return total
    return min(total, math.ceil(total / (1 + (count - 1) * (1 - overlap))))


def plan_tiles(width, height, levels=1, overlap=0.25):
    """Frozen full-frame plus overlapping tile schedule.

    Level k adds an aspect-matched grid whose tiles are cropped at source resolution, so a smaller
    tile reaches the model at higher magnification. Adjacent tiles share `overlap` of their length
    so an object on a tile border is whole in at least one view. The plan depends only on image
    dimensions and frozen parameters, never on image content.
    """
    if levels < 0 or not 0 <= overlap < 1:
        raise ValueError('invalid_tile_plan')
    jobs = [{'id': 'full', 'bbox_xyxy': [0.0, 0.0, float(width), float(height)], 'level': 0}]
    for level in range(1, levels + 1):
        columns, rows = grid_shape(width, height, level)
        tile_w = _tile_span(width, columns, overlap)
        tile_h = _tile_span(height, rows, overlap)
        for row, y in enumerate(_origins(height, tile_h, rows)):
            for column, x in enumerate(_origins(width, tile_w, columns)):
                jobs.append({'id': f'L{level}-r{row}c{column}', 'level': level,
                             'bbox_xyxy': [float(x), float(y), float(x + tile_w), float(y + tile_h)]})
    return jobs


def covered_fraction(jobs, width, height):
    """Fraction of the image inside at least one job, by exact pixel-grid union."""
    if not jobs:
        return 0.0
    grid = np.zeros((int(height), int(width)), dtype=bool)
    for job in jobs:
        x1, y1, x2, y2 = (int(round(v)) for v in job['bbox_xyxy'])
        grid[y1:y2, x1:x2] = True
    return float(grid.mean())


def ios(a, b):
    """Intersection over the smaller box.

    A detection truncated at a tile border overlaps its whole-object counterpart with low IoU but
    high IoS, so sliced-inference merging needs this rather than IoU (the SAHI matching rule).
    """
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller else 0.0
