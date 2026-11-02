"""Compatibility helpers for annotation coordinates."""
import math


def rectangle_points(points):
    """Labelme rectangles use two opposite corners; some sources store all four."""
    values = [[float(x), float(y)] for x, y in points]
    if len(values) < 2 or not all(math.isfinite(v) for p in values for v in p):
        raise ValueError('矩形坐标无效')
    return [[min(p[0] for p in values), min(p[1] for p in values)],
            [max(p[0] for p in values), max(p[1] for p in values)]]


def normalize_rectangles(doc):
    for shape in doc.get('shapes') or []:
        if shape.get('shape_type') == 'rectangle' and len(shape.get('points') or []) >= 2:
            shape['points'] = rectangle_points(shape['points'])
    return doc
