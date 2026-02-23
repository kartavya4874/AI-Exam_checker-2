"""
Retry utilities — exponential-backoff decorator for API calls.

Usage::

    @retry(max_retries=3, base_delay=1.0, exceptions=(openai.RateLimitError,))
    def call_api(...):
        ...
"""

import time
import functools
from typing import Callable, Tuple, Type

from exam_checker.utils.logger import get_logger

log = get_logger(__name__)


def retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    backoff_factor: float = 2.0,
) -> Callable:
    """
    Decorator that retries a function on specified exceptions using
    exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before the first retry.
        exceptions: Tuple of exception classes that trigger a retry.
        backoff_factor: Multiplier applied to the delay after each failure.

    Returns:
        Decorated function with retry behaviour.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception: Exception | None = None
            for attempt in range(1, max_retries + 2):  # +1 for initial try
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt > max_retries:
                        log.error(
                            "Function %s failed after %d retries: %s",
                            func.__name__,
                            max_retries,
                            exc,
                        )
                        raise
                    log.warning(
                        "Function %s attempt %d/%d failed (%s). "
                        "Retrying in %.1fs…",
                        func.__name__,
                        attempt,
                        max_retries + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            # Should not reach here, but just in case
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
