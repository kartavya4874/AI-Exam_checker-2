"""
Retry utilities with exponential backoff decorator.

Used primarily for OpenAI API calls and other network operations
that may experience transient failures.
"""

import time
import functools
import logging
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds (doubles each retry).
        max_delay: Maximum delay cap in seconds.
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        Decorated function with retry logic.

    Example::

        @retry_with_backoff(max_retries=3, exceptions=(openai.RateLimitError,))
        def call_gpt4o(prompt):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_retries:
                        logger.error(
                            "Function %s failed after %d retries: %s",
                            func.__name__,
                            max_retries,
                            str(exc),
                        )
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        "Attempt %d/%d for %s failed (%s). "
                        "Retrying in %.1fs...",
                        attempt + 1,
                        max_retries + 1,
                        func.__name__,
                        str(exc),
                        delay,
                    )
                    time.sleep(delay)
            raise last_exception  # Should never reach here

        return wrapper

    return decorator


class TokenBucket:
    """
    Simple token bucket rate limiter for API calls.

    Ensures we don't exceed a certain number of requests per minute.
    """

    def __init__(self, tokens_per_minute: int = 60):
        """
        Initialize the token bucket.

        Args:
            tokens_per_minute: Maximum number of tokens (requests) per minute.
        """
        self.capacity: int = tokens_per_minute
        self.tokens: float = float(tokens_per_minute)
        self.refill_rate: float = tokens_per_minute / 60.0  # tokens per second
        self.last_refill: float = time.time()
        self._lock_placeholder = None  # Use threading.Lock() in threaded context

    def acquire(self, timeout: float = 120.0) -> bool:
        """
        Acquire a token, blocking until one is available.

        Args:
            timeout: Maximum seconds to wait for a token.

        Returns:
            True if token acquired, False if timed out.
        """
        start = time.time()
        while True:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            elapsed = time.time() - start
            if elapsed >= timeout:
                logger.warning(
                    "TokenBucket: timed out waiting for token after %.1fs",
                    elapsed,
                )
                return False
            # Sleep until at least one token would be available
            sleep_time = min((1.0 - self.tokens) / self.refill_rate, 1.0)
            time.sleep(sleep_time)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
