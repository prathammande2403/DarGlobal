"""OpenAI-compatible completion clients for OpenRouter and Groq."""

import logging
import os
import time
from typing import Any, Optional

from openai import APIError, APIStatusError, OpenAI

from scraper.config import config as app_config

try:
    from dotenv import load_dotenv

    # Load an optional provider-specific file after the default .env. This keeps
    # test credentials/settings separate from submission credentials.
    env_file = os.getenv("LLM_ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=True)
except ImportError:
    pass

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_TIMEOUT = 30.0
RETRYABLE_STATUS_CODES = {404, 429}
BACKOFF_SECONDS = (1.0, 3.0)
DEFAULT_FALLBACK_MODELS = (
    "liquid/lfm-3-1.7b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)


class OpenRouterClient:
    """OpenAI-compatible client targeting OpenRouter."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        fallback_models: Optional[list[str]] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or app_config.openrouter_api_key or ""
        self.model = model or os.getenv("OPENROUTER_MODEL") or app_config.openrouter_model or "google/gemini-2.0-flash-lite:free"
        configured_fallbacks = os.getenv("OPENROUTER_FALLBACK_MODELS", "")
        parsed_fallbacks = [item.strip() for item in configured_fallbacks.split(",") if item.strip()]
        self.fallback_models = (
            fallback_models
            if fallback_models is not None
            else parsed_fallbacks or list(DEFAULT_FALLBACK_MODELS)
        )
        self.timeout = timeout
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _response_body(error: APIStatusError) -> str:
        try:
            return error.response.text
        except Exception:
            return str(getattr(error, "body", ""))

    @staticmethod
    def _retryable(status_code: int) -> bool:
        return status_code in RETRYABLE_STATUS_CODES or 500 <= status_code <= 599

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 500,
        **kwargs: Any,
    ) -> str:
        if not self.api_key or self.api_key == "your_openrouter_api_key_here":
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")

        last_error: Optional[Exception] = None
        for model in [self.model, *self.fallback_models]:
            for attempt in range(len(BACKOFF_SECONDS) + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    content = response.choices[0].message.content or ""
                    if not content.strip():
                        finish_reason = getattr(response.choices[0], "finish_reason", None)
                        try:
                            raise RuntimeError(
                                f"OpenRouter model {model} returned empty content (finish_reason={finish_reason})."
                            )
                        except RuntimeError:
                            logger.exception("Empty OpenRouter completion; trying the next fallback model")
                        last_error = RuntimeError(
                            f"OpenRouter model {model} returned empty content (finish_reason={finish_reason})."
                        )
                        break
                    logger.info("OpenRouter answered with model %s", model)
                    return content.strip()
                except APIStatusError as err:
                    status_code = err.status_code
                    body = self._response_body(err)
                    logger.exception(
                        "OpenRouter request failed: model=%s status=%s response_body=%s attempt=%d",
                        model,
                        status_code,
                        body,
                        attempt + 1,
                    )
                    last_error = err
                    if not self._retryable(status_code):
                        raise
                    if attempt < len(BACKOFF_SECONDS):
                        delay = BACKOFF_SECONDS[attempt]
                        logger.warning("Retrying model %s in %.1f seconds", model, delay)
                        time.sleep(delay)
                        continue
                    break
                except APIError:
                    logger.exception("OpenRouter transport/API error: model=%s attempt=%d", model, attempt + 1)
                    raise
                except Exception:
                    logger.exception("Unexpected OpenRouter error: model=%s attempt=%d", model, attempt + 1)
                    raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("No OpenRouter model was configured.")


class GroqClient(OpenRouterClient):
    """OpenAI-compatible client targeting Groq for testing."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.fallback_models = []
        self.timeout = timeout
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=GROQ_BASE_URL,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0,
                 max_tokens: int = 500, **kwargs: Any) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError(f"Groq model {self.model} returned empty content.")
        return content.strip()


_DEFAULT_CLIENT: Optional[OpenRouterClient] = None


def get_llm_client() -> OpenRouterClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
        if provider == "groq":
            _DEFAULT_CLIENT = GroqClient()
        elif provider == "openrouter":
            _DEFAULT_CLIENT = OpenRouterClient()
        else:
            raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")
    return _DEFAULT_CLIENT


def call_llm(
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> str:
    """Invoke the default OpenRouter client."""
    return get_llm_client().complete(messages, temperature=temperature, max_tokens=max_tokens)
