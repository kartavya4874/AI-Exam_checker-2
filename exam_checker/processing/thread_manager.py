"""
Thread Manager — Manages parallel execution of the processing pipeline.

Uses a ThreadPoolExecutor to process multiple students or multiple 
courses simultaneously while respecting resource limits.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Dict, Any, Optional

from exam_checker.config import MAX_THREADS
from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

class ThreadManager:
    """
    Manages concurrency for the exam checker system.
    """

    def __init__(self, max_workers: int = MAX_THREADS):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}
        self._lock = threading.Lock()
        log.info("ThreadManager initialized with %d workers", max_workers)

    def run_parallel(self, func: Callable, items: List[Any], *args, **kwargs) -> List[Any]:
        """
        Run a function in parallel over a list of items.
        """
        futures = {self.executor.submit(func, item, *args, **kwargs): item for item in items}
        results = []
        
        for future in as_completed(futures):
            item = futures[future]
            try:
                data = future.result()
                results.append(data)
                log.info("Task completed for item: %s", item)
            except Exception as e:
                log.error("Task failed for item %s: %s", item, e)
                results.append(None)
                
        return results

    def shutdown(self, wait=True):
        """Shut down the executor."""
        log.info("Shutting down ThreadManager...")
        self.executor.shutdown(wait=wait)
