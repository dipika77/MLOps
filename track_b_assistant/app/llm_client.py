"""
LLM client with retry + fallback.

Groq and vLLM both expose an OpenAI-compatible API, so we can use the
`openai` python package for both and just change base_url.

Order of providers we try:
    1. Groq - PRIMARY_MODEL   (openai/gpt-oss-120b)
    2. Groq - FALLBACK_MODEL  (llama-3.1-8b-instant, smaller and has separate rate limits)
    3. Local vLLM server      (only if VLLM_BASE_URL is set)

Each provider is retried a few times with exponential backoff for temporary
errors (rate limit, timeout, connection error, 5xx). If it still fails we move
to the next provider.
"""
import asyncio
import logging

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app import config

logger = logging.getLogger(__name__)

# only these errors are worth retrying. For example a wrong API key (401) will
# fail again, so we skip directly to the next provider.
RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)

# limit how many LLM calls run at the same time (protects us from hitting provider rate limits)
llm_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_LLM_CALLS)


class AllProvidersFailedError(Exception):
    pass


def build_providers():
    providers = []
    if config.GROQ_API_KEY:
        # max_retries=0 because we do our own retries with tenacity
        groq_client = AsyncOpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )
        providers.append({"name": f"groq/{config.PRIMARY_MODEL}", "client": groq_client, "model": config.PRIMARY_MODEL})
        providers.append({"name": f"groq/{config.FALLBACK_MODEL}", "client": groq_client, "model": config.FALLBACK_MODEL})
    else:
        logger.warning("GROQ_API_KEY is not set, Groq provider is disabled")

    if config.VLLM_BASE_URL:
        vllm_client = AsyncOpenAI(
            api_key="not-needed",
            base_url=config.VLLM_BASE_URL,
            timeout=config.LLM_TIMEOUT_SECONDS * 2,  # local CPU/GPU model can be slower
            max_retries=0,
        )
        providers.append({"name": f"vllm/{config.VLLM_MODEL}", "client": vllm_client, "model": config.VLLM_MODEL})

    return providers


PROVIDERS = build_providers()


@retry(
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _call_with_retry(client, **kwargs):
    return await client.chat.completions.create(**kwargs)


async def chat_completion(messages, tools=None, json_mode=False, temperature=None, top_p=None):
    """
    Sends messages to the LLM. Returns (response, provider_name).
    Raises AllProvidersFailedError if every provider fails.
    """
    if not PROVIDERS:
        raise AllProvidersFailedError("No LLM provider configured. Set GROQ_API_KEY or VLLM_BASE_URL.")

    last_error = None
    for provider in PROVIDERS:
        kwargs = {
            "model": provider["model"],
            "messages": messages,
            "temperature": config.TEMPERATURE if temperature is None else temperature,
            "top_p": config.TOP_P if top_p is None else top_p,
            "max_tokens": config.MAX_TOKENS,
        }
        # reasoning_effort only works for reasoning models (gpt-oss), llama models reject it
        if "gpt-oss" in provider["model"]:
            kwargs["reasoning_effort"] = config.REASONING_EFFORT
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            async with llm_semaphore:
                response = await _call_with_retry(provider["client"], **kwargs)
            return response, provider["name"]
        except Exception as e:
            logger.error("Provider %s failed: %s: %s", provider["name"], type(e).__name__, e)
            last_error = e

    raise AllProvidersFailedError(f"All LLM providers failed. Last error: {last_error}")
