"""
Institutional-grade retry utilities using tenacity.
Used across the trading platform for API resilience.
"""
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging
import httpx
import ccxt

logger = logging.getLogger("RetryHandler")

# Common exceptions to retry on
RETRYABLE_EXCEPTIONS = (
    httpx.HTTPError,
    httpx.ConnectError,
    httpx.TimeoutException,
    ccxt.NetworkError,
    ccxt.ExchangeError,
    ConnectionError,
    TimeoutError,
)

def get_api_retry_decorator(max_attempts: int = 4):
    """Returns a robust retry decorator for external APIs."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.8, min=0.5, max=8),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )