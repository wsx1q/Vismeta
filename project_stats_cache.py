"""Serve the last complete counts while one background worker refreshes them."""
import json
import logging
import threading
import time
from pathlib import Path


class StatsCache:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.rows = {}
        self.updated = 0
        self.running = False
        self.generation = 0
        self.completed_generation = 0
        self.retry_after = 0
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            self.rows = data['rows']
            self.updated = float(data['updated'])
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def invalidate(self):
        with self.lock:
            self.generation += 1

    def snapshot(self, projects, compute):
        with self.lock:
            now = time.time()
            needed = (now - self.updated >= 30 or self.generation != self.completed_generation
                      or any(p['id'] not in self.rows for p in projects))
            if needed and not self.running and now >= self.retry_after:
                self.running = True
                threading.Thread(target=self._refresh, args=(list(projects), compute, self.generation),
                                 daemon=True, name='project-statistics').start()
            return dict(self.rows), self.running

    def _refresh(self, projects, compute, generation):
        try:
            result = {p['id']: compute(p) for p in projects}
            updated = time.time()
            payload = json.dumps({'rows': result, 'updated': updated}, ensure_ascii=False)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(payload, encoding='utf-8')
            tmp.replace(self.path)
            with self.lock:
                self.rows = result
                self.updated = updated
                self.completed_generation = generation
        except Exception:
            logging.exception('Project statistics refresh failed; keeping previous counts')
        finally:
            with self.lock:
                self.retry_after = time.time() + 2
                self.running = False


_CACHES = {}
_LOCK = threading.Lock()


def cache(data_root):
    key = str(Path(data_root).resolve())
    with _LOCK:
        if key not in _CACHES:
            _CACHES[key] = StatsCache(Path(data_root) / '.project-stats.json')
        return _CACHES[key]


def invalidate(data_root):
    # No need to load or create a cache for clients that do not use it.
    with _LOCK:
        value = _CACHES.get(str(Path(data_root).resolve()))
    if value is not None:
        value.invalidate()
