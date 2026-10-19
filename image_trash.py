"""Recoverable removal of project images and their annotations."""
import json
import threading
import time
import uuid
from pathlib import Path

LOCK = threading.Lock()
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _root(project):
    return Path(project['images']).parent / '.trash'


def _write(folder, row):
    tmp = folder / 'manifest.json.tmp'
    tmp.write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
    tmp.replace(folder / 'manifest.json')


def remove(project, names):
    if not isinstance(names, list) or not names or len(names) > 50000:
        raise ValueError('请选择需要删除的图片')
    if any(not isinstance(n, str) or n != Path(n).name or '/' in n or '\\' in n or '\x00' in n for n in names):
        raise ValueError('图片名称无效')
    names = list(dict.fromkeys(names))
    with LOCK:
        images = Path(project['images']).resolve()
        annotations = Path(project['annotations']).resolve()
        for name in names:
            p = images / name
            if p.suffix.lower() not in EXTS or p.is_symlink() or not p.is_file():
                raise ValueError('图片不存在：' + name)
        selected = set(names)
        remaining = {p.stem for p in images.iterdir() if p.name not in selected and p.suffix.lower() in EXTS}
        seen = set()
        rows = []
        for name in names:
            stem = Path(name).stem
            ann = annotations / (stem + '.json')
            take = stem not in remaining and stem not in seen and ann.is_file() and not ann.is_symlink()
            rows.append({'name': name, 'annotation': ann.name if take else None})
            seen.add(stem)
        batch = uuid.uuid4().hex
        folder = _root(project) / batch
        (folder/'images').mkdir(parents=True)
        (folder/'annotations').mkdir()
        row = {'id': batch, 'created': time.time(), 'state': 'pending', 'items': rows}
        _write(folder, row)
        moved = []
        try:
            for item in rows:
                pairs = [(images/item['name'], folder/'images'/item['name'])]
                if item['annotation']:
                    pairs.append((annotations/item['annotation'], folder/'annotations'/item['annotation']))
                for src, dst in pairs:
                    src.replace(dst); moved.append((src, dst))
            row['state'] = 'deleted'; _write(folder, row)
        except Exception:
            for src, dst in reversed(moved):
                if dst.exists() and not src.exists(): dst.replace(src)
            raise
        return {'count': len(rows), 'batch': batch}


def list_batches(project):
    result = []
    with LOCK:
        for p in sorted(_root(project).glob('*/manifest.json'), reverse=True):
            row = json.loads(p.read_text(encoding='utf-8'))
            names = [item['name'] for item in row['items'] if (p.parent/'images'/item['name']).is_file() or (item['annotation'] and (p.parent/'annotations'/item['annotation']).is_file())]
            if names:
                result.append({'id': row['id'], 'created': row['created'], 'count': len(names), 'names': names[:5]})
    return sorted(result, key=lambda row: row['created'], reverse=True)


def restore(project, batch):
    if not isinstance(batch, str) or len(batch) != 32 or any(c not in '0123456789abcdef' for c in batch):
        raise ValueError('回收记录无效')
    with LOCK:
        folder = _root(project)/batch
        if not (folder/'manifest.json').is_file(): raise ValueError('回收记录不存在')
        row = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        pairs = []
        for item in row['items']:
            for subdir, name, root in [('images', item['name'], Path(project['images'])),
                                       ('annotations', item['annotation'], Path(project['annotations']))]:
                if not name: continue
                src, dst = folder/subdir/name, root/name
                if not src.is_file(): continue
                if dst.exists(): raise ValueError('存在同名文件，无法覆盖恢复：' + name)
                pairs.append((src, dst))
        if not pairs: raise ValueError('该批图片已经恢复')
        done = []
        try:
            for src, dst in pairs:
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst); done.append((src, dst))
            row['state'] = 'restored'; _write(folder, row)
        except Exception:
            for src, dst in reversed(done):
                if dst.exists() and not src.exists(): dst.replace(src)
            raise
        return {'count': sum(dst.parent == Path(project['images']) for _, dst in pairs)}
