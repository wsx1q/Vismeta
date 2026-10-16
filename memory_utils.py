"""Return freed native CPU allocations to Linux without flushing on every request."""
import ctypes
import sys
import threading
import time

_lock = threading.Lock()
_last_trim = 0.0
_trim = None
if sys.platform.startswith('linux'):
    try:
        _trim = ctypes.CDLL(None).malloc_trim
        _trim.argtypes = [ctypes.c_size_t]
        _trim.restype = ctypes.c_int
    except (AttributeError, OSError):
        pass


def trim_cpu_memory(force=False):
    global _last_trim
    if _trim is None or not _lock.acquire(blocking=False):
        return
    try:
        now = time.monotonic()
        if force or now - _last_trim >= 15:
            _trim(0)
            _last_trim = now
    finally:
        _lock.release()
