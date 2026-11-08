"""Process-local Triton cache for the competition sandbox."""

import os
import threading
import uuid
from pathlib import Path

from triton.runtime.cache import FileCacheManager


_NONCE = uuid.uuid4().hex
_LOCK = threading.RLock()


class PrivateCache(FileCacheManager):
    def __init__(self, key, override=False, dump=False):
        super().__init__(key, override=override, dump=dump)
        if not override and not dump:
            self.cache_dir = str(Path(self.cache_dir) / f"{os.getpid()}-{_NONCE}")
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def has_file(self, filename):
        with _LOCK:
            return super().has_file(filename)

    def put(self, data, filename, binary=True):
        target = Path(self._make_path(filename))
        with _LOCK:
            if isinstance(data, bytes):
                target.write_bytes(data)
            else:
                target.write_text(str(data), encoding="utf-8")
        return str(target)
