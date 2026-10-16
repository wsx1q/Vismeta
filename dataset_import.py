"""Import image trees with Labelme or YOLO detection/segmentation sidecars."""
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import shutil
from pathlib import Path

import yaml
from PIL import Image

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def class_names(root):
    root = Path(root)
    candidates = [root / n for n in ('data.yaml', 'data.yml', 'dataset.yaml', 'dataset.yml', 'classes.txt', 'obj.names')]
    candidates += sorted(root.glob('*.yaml')) + sorted(root.glob('*.yml'))
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        if path.suffix in {'.yaml', '.yml'}:
            doc = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
            names = doc.get('names') if isinstance(doc, dict) else None
            if isinstance(names, list):
                return {i: str(n) for i, n in enumerate(names)}
            if isinstance(names, dict):
                return {int(i): str(n) for i, n in names.items()}
        else:
            return dict(enumerate(path.read_text(encoding='utf-8-sig').splitlines()))
    return None


def sidecar(image, suffix, boundary):
    candidates = [image.with_suffix(suffix)]
    # Support images/train/foo.jpg -> labels/train/foo.txt and train/images/foo.jpg.
    for parent in image.parents:
        if parent.name.lower() == 'images':
            candidates.append((parent.parent / 'labels' / image.relative_to(parent)).with_suffix(suffix))
        if parent == boundary:
            break
    candidates.append(boundary / 'annotations' / (image.stem + suffix))
    return next((p for p in candidates if p.is_file()), None)


def yolo_shapes(path, width, height, names):
    shapes = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            parts = line.split()
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            if names is None or cls not in names or not names[cls].strip():
                raise ValueError('缺少类别表或类别编号不存在，请提供 data.yaml / classes.txt')
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords):
                raise ValueError('YOLO 坐标必须是 0 到 1 的有限数值')
            if len(coords) == 4:
                x, y, w, h = coords
                if w <= 0 or h <= 0:
                    raise ValueError('框宽高必须大于 0')
                points = [[max(0, (x-w/2)*width), max(0, (y-h/2)*height)],
                          [min(width, (x+w/2)*width), min(height, (y+h/2)*height)]]
                kind = 'rectangle'
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                points = [[coords[i]*width, coords[i+1]*height] for i in range(0, len(coords), 2)]
                kind = 'polygon'
            else:
                raise ValueError('仅支持 YOLO 检测框和分割多边形')
            shapes.append({'label': names[cls], 'points': points, 'shape_type': kind, 'group_id': None, 'flags': {}})
        except (ValueError, IndexError) as exc:
            raise ValueError(f'{path.name} 第 {line_no} 行：{exc}') from exc
    return shapes


def import_tree(source, project, cap=20000, progress=None):
    source = Path(source).resolve()
    boundary = source if source.is_dir() else source.parent
    images_root = Path(project['images']).resolve()
    annotations_root = Path(project['annotations'])
    images_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(phase='scanning', processed=0, total=0, count=0, skipped=0, shapes=0)
    images = [source] if source.is_file() else sorted(source.rglob('*'))
    images = [p for p in images if p.is_file() and p.suffix.lower() in EXTS and '__MACOSX' not in p.parts
              and images_root != p.parent.resolve() and images_root not in p.resolve().parents]
    total = min(len(images), cap)
    processed = 0
    if progress:
        progress(phase='importing', processed=0, total=total, count=0, skipped=0, shapes=0)
    added, errors, labels = [], [], set()
    annotated = objects = empty = 0
    cache = {}
    cache_lock = threading.Lock()
    # Allocate names before dispatch so equal stems across splits cannot race.
    used = {p.stem for root in (images_root, annotations_root) for p in root.iterdir() if p.is_file()}
    work = []
    for image in images[:cap]:
        stem, n = image.stem, 1
        while stem in used:
            n += 1
            stem = f'{image.stem}_{n}'
        used.add(stem)
        work.append((image, images_root / (stem + image.suffix.lower())))

    def import_one(image, dest):
        ann = sidecar(image, '.json', boundary)
        txt = sidecar(image, '.txt', boundary) if ann is None else None
        doc = None
        if ann or txt:
            with Image.open(image) as im:
                width, height = im.size
            if ann:
                doc = json.loads(ann.read_text(encoding='utf-8-sig'))
                if not isinstance(doc, dict) or not isinstance(doc.get('shapes'), list):
                    raise ValueError('JSON 不是 Labelme 标注')
                for shape in doc['shapes']:
                    kind = shape.get('shape_type', 'polygon')
                    points = shape.get('points', [])
                    if kind not in {'rectangle', 'polygon'} or len(points) < (2 if kind == 'rectangle' else 3):
                        raise ValueError(f'不支持的标注形状：{kind}')
                    if not str(shape.get('label', '')).strip() or not all(len(p) == 2 and all(math.isfinite(float(v)) for v in p) for p in points):
                        raise ValueError('标签或坐标无效')
                doc.pop('imageData', None)
            else:
                names = None
                for parent in image.parents:
                    with cache_lock:
                        if parent not in cache:
                            cache[parent] = class_names(parent)
                        parent_names = cache[parent]
                    if parent_names is not None:
                        names = parent_names
                        break
                    if parent == boundary:
                        break
                doc = {'shapes': yolo_shapes(txt, width, height, names), 'flags': {}}
            doc.update(version='2.4.0', imageData=None, imageWidth=width, imageHeight=height)
        # Exclusive creation also protects against another uploader writing this name.
        with dest.open('xb'):
            pass
        annotation = annotations_root / (dest.stem + '.json')
        try:
            shutil.copy2(image, dest)
            if doc is not None:
                doc['imagePath'] = dest.name
                annotation.write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
        except Exception:
            dest.unlink(missing_ok=True)
            annotation.unlink(missing_ok=True)
            raise
        return dest.name, doc

    workers = max(1, min(8, int(os.environ.get('IMPORT_WORKERS', '4'))))
    work_iter = iter(work)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='import-copy') as pool:
        pending = {}
        def submit_next():
            task = next(work_iter, None)
            if task is not None:
                image, dest = task
                pending[pool.submit(import_one, image, dest)] = image
        for _ in range(workers):
            submit_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                image = pending.pop(future)
                try:
                    name, doc = future.result()
                    added.append(name)
                    if doc is not None:
                        annotated += bool(doc['shapes'])
                        empty += not bool(doc['shapes'])
                        objects += len(doc['shapes'])
                        labels.update(s['label'] for s in doc['shapes'])
                except Exception as exc:
                    errors.append(f'{image.name}: {exc}')
                processed += 1
                if progress:
                    progress(phase='importing', processed=processed, total=total, count=len(added),
                             skipped=len(errors), shapes=objects, filename=image.name)
                submit_next()
    if len(images) > cap:
        errors.append(f'达到单次 {cap} 张上限，余下图片未导入')
    labels_path = Path(project['labels'])
    if labels:
        old = json.loads(labels_path.read_text(encoding='utf-8')) if labels_path.exists() else {'labels': []}
        existing = {s['name'] for s in old['labels']}
        old['labels'].extend({'name': name, 'locked': False} for name in sorted(labels - existing))
        labels_path.write_text(json.dumps(old, ensure_ascii=False), encoding='utf-8')
    return {'count': len(added), 'annotated': annotated, 'shapes': objects, 'empty_annotations': empty,
            'skipped': len(errors), 'errors': errors[:30], 'names': added[:30]}


def extract_archive(path, root):
    """Extract only dataset files; reject traversal and links."""
    import tarfile
    import zipfile
    root = Path(root).resolve()
    allowed = EXTS | {'.txt', '.names', '.json', '.yaml', '.yml'}

    def dest(name):
        name = name.replace('\\', '/')
        p = (root / name).resolve()
        if p == root or root not in p.parents or ':' in name or '__MACOSX' in p.parts:
            return None
        return p if p.suffix.lower() in allowed else None

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as arc:
            for item in arc.infolist():
                p = dest(item.filename)
                if p is None or item.is_dir() or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                with arc.open(item) as src, p.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as arc:
            for item in arc:
                p = dest(item.name)
                if p is None or not item.isfile():
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                with arc.extractfile(item) as src, p.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    else:
        raise ValueError('仅支持 zip / tar 压缩包')
