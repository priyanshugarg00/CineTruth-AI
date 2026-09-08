"""Shared Gemini reliability helpers for CineTruth AI.

This module keeps transient Gemini errors (notably HTTP 429/503) from
immediately failing a scan. Calls use exponential backoff and can fall back to
other configured stable Flash models.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

from config import Config

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
_AUTH_CODES = {401, 403}


@dataclass
class GeminiCallMeta:
    model_used: str | None
    models_tried: list[str]
    requests_used: int
    retries_used: int


class GeminiRequestFailure(RuntimeError):
    """Raised after Gemini retries/fallbacks are exhausted."""

    def __init__(
        self,
        *,
        kind: str,
        public_message: str,
        technical_error: str,
        requests_used: int,
        models_tried: list[str],
    ) -> None:
        super().__init__(public_message)
        self.kind = kind
        self.public_message = public_message
        self.technical_error = technical_error
        self.requests_used = requests_used
        self.models_tried = models_tried


def _error_text(exc: Exception) -> str:
    return str(exc or "")


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None and str(value).isdigit():
                return int(value)
        except Exception:
            pass

    # Google SDK error strings commonly contain the HTTP code near the start.
    match = re.search(r"\b(400|401|403|404|408|409|429|500|502|503|504)\b", _error_text(exc))
    return int(match.group(1)) if match else None


def _is_retryable(exc: Exception) -> bool:
    code = _status_code(exc)
    if code in _RETRYABLE_CODES:
        return True
    text = _error_text(exc).lower()
    markers = (
        "unavailable",
        "resource_exhausted",
        "too many requests",
        "rate limit",
        "temporarily unavailable",
        "high demand",
        "deadline exceeded",
        "timeout",
        "timed out",
    )
    return any(marker in text for marker in markers)


def _is_auth_error(exc: Exception) -> bool:
    code = _status_code(exc)
    if code in _AUTH_CODES:
        return True
    text = _error_text(exc).lower()
    return any(
        marker in text
        for marker in (
            "api key not valid",
            "invalid api key",
            "permission_denied",
            "unauthenticated",
        )
    )


def _is_model_error(exc: Exception) -> bool:
    code = _status_code(exc)
    text = _error_text(exc).lower()
    if code == 404 and "model" in text:
        return True
    return any(
        marker in text
        for marker in (
            "model is no longer available",
            "model not found",
            "no longer available to new users",
            "unsupported model",
        )
    )


def _is_daily_quota_error(exc: Exception) -> bool:
    text = _error_text(exc).lower()
    daily_markers = (
        "generaterequestsperdayperprojectpermodel",
        "generate_content_free_tier_requests",
        "per day",
        "daily quota",
    )
    return "quota" in text and any(marker in text for marker in daily_markers)


def model_candidates(primary: str | None = None) -> list[str]:
    """Return a deduplicated ordered primary + fallback model list."""
    candidates: list[str] = []
    for value in [primary or Config.GEMINI_MODEL, *Config.GEMINI_FALLBACK_MODELS]:
        value = str(value or "").strip()
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def generate_content_with_fallback(
    client: Any,
    *,
    contents: Any,
    config: Any | None = None,
    primary_model: str | None = None,
) -> tuple[Any, GeminiCallMeta]:
    """Generate content with retry/backoff and model fallback.

    `requests_used` counts every generate_content attempt, including failed
    attempts, so the UI telemetry reflects what actually happened.
    """
    if client is None:
        raise GeminiRequestFailure(
            kind="CONFIG_ERROR",
            public_message="GEMINI_API_KEY is not configured.",
            technical_error="Gemini client is not configured.",
            requests_used=0,
            models_tried=[],
        )

    models = model_candidates(primary_model)
    if not models:
        raise GeminiRequestFailure(
            kind="CONFIG_ERROR",
            public_message="No Gemini model is configured.",
            technical_error="GEMINI_MODEL and GEMINI_FALLBACK_MODELS are empty.",
            requests_used=0,
            models_tried=[],
        )

    attempts_per_model = max(1, int(Config.GEMINI_RETRY_ATTEMPTS))
    initial_delay = max(0.0, float(Config.GEMINI_RETRY_INITIAL_DELAY))
    max_delay = max(initial_delay, float(Config.GEMINI_RETRY_MAX_DELAY))

    requests_used = 0
    retries_used = 0
    tried: list[str] = []
    last_exc: Exception | None = None
    saw_daily_quota = False

    for model in models:
        tried.append(model)

        for attempt_index in range(attempts_per_model):
            requests_used += 1
            try:
                kwargs = {"model": model, "contents": contents}
                if config is not None:
                    kwargs["config"] = config
                response = client.models.generate_content(**kwargs)
                return response, GeminiCallMeta(
                    model_used=model,
                    models_tried=tried.copy(),
                    requests_used=requests_used,
                    retries_used=retries_used,
                )
            except Exception as exc:  # SDK exposes multiple exception classes across versions.
                last_exc = exc
                saw_daily_quota = saw_daily_quota or _is_daily_quota_error(exc)

                if _is_auth_error(exc):
                    raise GeminiRequestFailure(
                        kind="AUTH_ERROR",
                        public_message=(
                            "Gemini authorization failed. Check GEMINI_API_KEY and the API/project permissions."
                        ),
                        technical_error=_error_text(exc),
                        requests_used=requests_used,
                        models_tried=tried.copy(),
                    ) from exc

                # A removed/unsupported model should immediately move to the next fallback.
                if _is_model_error(exc):
                    logger.warning("Gemini model %s is unavailable; trying fallback. Error: %s", model, exc)
                    break

                if not _is_retryable(exc):
                    raise GeminiRequestFailure(
                        kind="API_ERROR",
                        public_message="Gemini analysis could not be completed because the API returned an unexpected error.",
                        technical_error=_error_text(exc),
                        requests_used=requests_used,
                        models_tried=tried.copy(),
                    ) from exc

                # Retry the same model before moving to a fallback.
                if attempt_index < attempts_per_model - 1:
                    delay = min(max_delay, initial_delay * (2 ** attempt_index))
                    if delay > 0:
                        delay += random.uniform(0, min(0.5, delay * 0.15))
                    retries_used += 1
                    logger.warning(
                        "Gemini transient error on %s (attempt %s/%s). Retrying in %.2fs: %s",
                        model,
                        attempt_index + 1,
                        attempts_per_model,
                        delay,
                        exc,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    continue

                logger.warning("Gemini model %s still unavailable after retries; trying fallback.", model)

    technical_error = _error_text(last_exc) if last_exc else "Unknown Gemini error"
    if saw_daily_quota:
        kind = "QUOTA_EXCEEDED"
        public_message = (
            "Gemini quota is currently exhausted for the available model/project. "
            "Automatic retries and fallback models were attempted, but no AI score could be generated."
        )
    else:
        kind = "TEMPORARILY_UNAVAILABLE"
        public_message = (
            "Gemini is temporarily busy or unavailable. CineTruth automatically retried the request and tried fallback "
            "models, but the service is still unavailable. Please retry the scan shortly."
        )

    raise GeminiRequestFailure(
        kind=kind,
        public_message=public_message,
        technical_error=technical_error,
        requests_used=requests_used,
        models_tried=tried,
    ) from last_exc


def run_transient_operation(
    operation,
    *,
    operation_name: str = "Gemini operation",
    attempts: int | None = None,
):
    """Retry non-generation Gemini operations such as file upload/get."""
    max_attempts = max(1, int(attempts or Config.GEMINI_FILE_RETRY_ATTEMPTS))
    initial_delay = max(0.0, float(Config.GEMINI_RETRY_INITIAL_DELAY))
    max_delay = max(initial_delay, float(Config.GEMINI_RETRY_MAX_DELAY))
    last_exc: Exception | None = None

    for attempt_index in range(max_attempts):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc

            if _is_auth_error(exc):
                raise GeminiRequestFailure(
                    kind="AUTH_ERROR",
                    public_message="Gemini authorization failed. Check GEMINI_API_KEY and the API/project permissions.",
                    technical_error=_error_text(exc),
                    requests_used=0,
                    models_tried=[],
                ) from exc

            if not _is_retryable(exc):
                raise

            if attempt_index >= max_attempts - 1:
                raise GeminiRequestFailure(
                    kind="TEMPORARILY_UNAVAILABLE",
                    public_message=(
                        "Gemini is temporarily busy while preparing the media. Automatic retries were attempted; "
                        "please retry the scan shortly."
                    ),
                    technical_error=_error_text(exc),
                    requests_used=0,
                    models_tried=[],
                ) from exc

            delay = min(max_delay, initial_delay * (2 ** attempt_index))
            if delay > 0:
                delay += random.uniform(0, min(0.5, delay * 0.15))
            logger.warning(
                "%s failed transiently (attempt %s/%s). Retrying in %.2fs: %s",
                operation_name,
                attempt_index + 1,
                max_attempts,
                delay,
                exc,
            )
            if delay > 0:
                time.sleep(delay)

    if last_exc:
        raise last_exc
    raise RuntimeError(f"{operation_name} failed.")
