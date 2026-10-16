"""On-demand project backups, queued after exports of moved datasets."""
import hashlib
import json
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='project-backup')
_LOCK = threading.Lock()
_JOBS = {}


def _folder(data_root, pid):
    return Path(data_root) / 'backups' / pid


def status(data_root, pid):
    key = (str(data_root), pid)
    with _LOCK:
        if key in _JOBS:
            return dict(_JOBS[key])
    path = _folder(data_root, pid) / 'status.json'
    if not path.exists():
        return {'status': 'none'}
    row = json.loads(path.read_text(encoding='utf-8'))
    if row.get('status') in {'queued', 'running'}:
        row.update(status='failed', error='服务重启，备份已中断；重新导出可再次备份')
    return row


def schedule(data_root, project):
    root = Path(project['images']).parent
    if not (root / 'move-import.jsonl').is_file():
        return None
    key = (str(data_root), project['id'])
    with _LOCK:
        existing = _JOBS.get(key)
        if existing and existing['status'] in {'queued', 'running'}:
            return dict(existing)
        row = {'id': uuid.uuid4().hex, 'status': 'queued', 'done': 0, 'total': 0, 'updated': time.time()}
        _JOBS[key] = row
    folder = _folder(data_root, project['id'])
    folder.mkdir(parents=True, exist_ok=True)

    last_saved = 0

    def update(**values):
        nonlocal last_saved
        with _LOCK:
            row.update(values, updated=time.time())
            data = json.dumps(row, ensure_ascii=False)
        now = time.monotonic()
        if 'status' not in values and now - last_saved < 1:
            return
        last_saved = now
        tmp = folder / 'status.json.tmp'
        tmp.write_text(data, encoding='utf-8')
        tmp.replace(folder / 'status.json')

    def worker():
        partial = folder / (row['id'] + '.zip.part')
        try:
            files = sorted(p for p in root.rglob('*') if p.is_file())
            fingerprint = hashlib.sha256(json.dumps(project, sort_keys=True).encode())
            total = 0
            for p in files:
                st = p.stat()
                total += st.st_size
                fingerprint.update(f'{p.relative_to(root)}:{st.st_size}:{st.st_mtime_ns}\n'.encode())
            digest = fingerprint.hexdigest()
            previous_path = folder / 'latest.json'
            if previous_path.is_file():
                previous = json.loads(previous_path.read_text(encoding='utf-8'))
                archive = folder / Path(previous.get('archive', '')).name
                if previous.get('fingerprint') == digest and archive.is_file():
                    update(status='complete', done=total, total=total, archive=str(archive), reused=True)
                    return
            update(status='running', total=total)
            done = 0
            # Images are already compressed. Store without recompression or RAM buffering.
            with zipfile.ZipFile(partial, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
                archive.writestr('project.json', json.dumps(project, ensure_ascii=False))
                for p in files:
                    archive.write(p, str(p.relative_to(root)))
                    done += p.stat().st_size
                    update(done=done, filename=p.name)
            final = folder / (row['id'] + '.zip')
            partial.replace(final)
            latest_tmp = folder / 'latest.json.tmp'
            latest_tmp.write_text(json.dumps({'fingerprint': digest, 'archive': final.name}), encoding='utf-8')
            latest_tmp.replace(previous_path)
            update(status='complete', archive=str(final), reused=False)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            update(status='failed', error=str(exc))
    update(status='queued')
    _POOL.submit(worker)
    return dict(row)
