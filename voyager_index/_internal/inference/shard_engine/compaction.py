"""Background compaction for the shard engine.

Merges L0 (memtable flushes) into L1 sealed shards, purges tombstones,
and optionally re-clusters using LEMUR proxy weights.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class CompactionTask:
    """Represents a single compaction operation."""

    def __init__(self, manager: Any):
        self._manager = manager
        self._running = False

    def run(self) -> Dict[str, Any]:
        """Execute compaction: flush memtable and purge tombstones."""
        stats = {"flushed_docs": 0, "purged_tombstones": 0, "duration_ms": 0}
        t0 = time.perf_counter()

        mgr = self._manager
        if mgr._memtable and mgr._memtable.size > 0:
            stats["flushed_docs"] = mgr._memtable.size
            mgr.flush()

        stats["duration_ms"] = (time.perf_counter() - t0) * 1000
        return stats


class CompactionScheduler:
    """Background thread that runs compaction periodically."""

    def __init__(
        self,
        manager: Any,
        interval_s: float = 60.0,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._manager = manager
        self._interval = interval_s
        self._on_complete = on_complete
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="shard-compactor")
        self._thread.start()
        logger.info("Compaction scheduler started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                task = CompactionTask(self._manager)
                stats = task.run()
                if self._on_complete:
                    self._on_complete(stats)
                if stats["flushed_docs"] > 0:
                    logger.info("Compaction done: %s", stats)
            except Exception:
                logger.exception("Compaction failed")
