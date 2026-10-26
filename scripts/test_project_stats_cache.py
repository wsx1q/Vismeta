import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from project_stats_cache import StatsCache


class StatsCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'stats.json'
        self.projects = [{'id': 'p1'}]
        self.counts = {'images': 10, 'annotated': 2, 'shapes': 5}

    def wait_idle(self, cache):
        deadline = time.monotonic() + 3
        while cache.running and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertFalse(cache.running)

    def test_slow_scan_does_not_block_requests_and_only_one_worker_runs(self):
        cache = StatsCache(self.path)
        entered, release = threading.Event(), threading.Event()
        calls = []
        def compute(project):
            calls.append(project)
            entered.set()
            release.wait(3)
            return self.counts
        try:
            self.assertEqual(cache.snapshot(self.projects, compute), ({}, True))
            self.assertTrue(entered.wait(1))
            for _ in range(20):
                self.assertEqual(cache.snapshot(self.projects, compute), ({}, True))
            self.assertEqual(len(calls), 1)
        finally:
            release.set()
            self.wait_idle(cache)
        self.assertEqual(cache.snapshot(self.projects, compute), ({'p1': self.counts}, False))

    def test_persistent_counts_survive_restart_and_failed_refresh(self):
        self.path.write_text(json.dumps({'rows': {'p1': self.counts}, 'updated': 0}))
        cache = StatsCache(self.path)
        def fail(project):
            raise OSError('disk temporarily unavailable')
        with patch('project_stats_cache.logging.exception'):
            self.assertEqual(cache.snapshot(self.projects, fail)[0], {'p1': self.counts})
            self.wait_idle(cache)
        self.assertEqual(StatsCache(self.path).rows['p1'], self.counts)

    def test_edit_during_refresh_is_not_lost_and_cache_is_atomic(self):
        cache = StatsCache(self.path)
        entered, release = threading.Event(), threading.Event()
        def compute(project):
            entered.set(); release.wait(3)
            return self.counts
        cache.snapshot(self.projects, compute)
        self.assertTrue(entered.wait(1))
        cache.invalidate()
        release.set(); self.wait_idle(cache)
        self.assertNotEqual(cache.generation, cache.completed_generation)
        cache.retry_after = 0
        changed = {**self.counts, 'images': 9}
        cache.snapshot(self.projects, lambda project: changed)
        self.wait_idle(cache)
        self.assertEqual(cache.generation, cache.completed_generation)
        self.assertEqual(StatsCache(self.path).rows['p1'], changed)
