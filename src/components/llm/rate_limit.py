"""Log selected response headers when an inference request is rate-limited."""

import logging

from aiohttp import ClientResponseError

logger = logging.getLogger(__name__)

RATE_LIMIT_HEADERS = (
    "x-request-id",
    "x-amzn-trace-id",
    "retry-after",
    "ratelimit",
    "ratelimit-policy",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
)


def log_provider_rate_limit(stage: str, error: Exception) -> None:
    if not isinstance(error, ClientResponseError) or error.status != 429:
        return

    headers = error.headers or {}
    fields = []
    for name in RATE_LIMIT_HEADERS:
        value = headers.get(name)
        if value:
            clean_value = " ".join(value.split())[:200]
            fields.append(f"{name}={clean_value}")

    logger.warning(
        "Inference HTTP 429 stage=%s %s",
        stage,
        " ".join(fields) or "no diagnostic headers returned",
    )

def retry_after_seconds(error: Exception, max_delay: int = 5) -> int | None:
    if not isinstance(error, ClientResponseError) or error.status != 429:
        return None

    value = (error.headers or {}).get("Retry-After", "").strip()
    if not value.isascii() or not value.isdecimal() or len(value) > 3:
        return None

    delay = int(value)
    return delay if delay <= max_delay else None