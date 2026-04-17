from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import time


if os.name == "nt":  # pragma: no cover - platform specific
    import msvcrt
else:  # pragma: no cover - platform specific
    import fcntl


class ProcessFileLock:
    def __init__(self, lock_file: Path, *, timeout_seconds: float = 10.0, poll_interval: float = 0.05) -> None:
        self._lock_file = lock_file
        self._timeout_seconds = timeout_seconds
        self._poll_interval = poll_interval

    @contextmanager
    def acquire(self):
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_file.open("a+b") as handle:
            deadline = time.monotonic() + self._timeout_seconds
            while True:
                try:
                    if os.name == "nt":  # pragma: no branch
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:  # pragma: no cover - platform specific
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Не удалось захватить lock-файл {self._lock_file}.")
                    time.sleep(self._poll_interval)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":  # pragma: no branch
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - platform specific
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
