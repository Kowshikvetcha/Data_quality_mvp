import json

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.ai_tools import CLEANING_TOOLS
from core.logger import get_logger
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TIMEOUT_SECONDS, OPENAI_MAX_RETRIES

logger = get_logger("ai_router")

# Cache decorator: use st.cache_resource in Streamlit, no-op otherwise (tests)
try:
    import streamlit as _st
    _cache_resource = _st.cache_resource
except Exception:
    _cache_resource = lambda f: f


@_cache_resource
def _get_client():
    if not OPENAI_API_KEY:
        raise ConnectionError(
            "OpenAI API key is not configured. "
            "Please set the OPENAI_API_KEY environment variable in your .env file."
        )
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def build_system_prompt(column_types: dict) -> str:
    cols = "\n".join([f"- {c} ({t})" for c, t in column_types.items()])
    return f"""
You are a data cleaning assistant.

Available columns:
{cols}

Rules:
- Use ONLY the tools provided
- Use ONLY column names from the list
- Resolve minor misspellings
- Do NOT invent columns or tools
- Return a tool call with valid JSON
"""


def _make_api_call(client, messages, tools):
    """Retryable OpenAI API call with exponential backoff for transient errors."""
    try:
        from openai import APITimeoutError, APIConnectionError, RateLimitError
        _retry_types = (APITimeoutError, APIConnectionError, RateLimitError)
    except ImportError:
        _retry_types = ()

    @retry(
        stop=stop_after_attempt(OPENAI_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_retry_types),
        reraise=True,
    )
    def _call():
        return client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            timeout=OPENAI_TIMEOUT_SECONDS,
        )

    return _call()


def route_user_request(user_message: str, column_types: dict):
    client = _get_client()
    logger.info("AI request: model=%s, message='%s'", OPENAI_MODEL, (user_message or "")[:100])

    try:
        response = _make_api_call(
            client,
            messages=[
                {"role": "system", "content": build_system_prompt(column_types or {})},
                {"role": "user", "content": user_message or ""}
            ],
            tools=CLEANING_TOOLS,
        )
    except Exception as e:
        logger.error("AI API error: %s: %s", type(e).__name__, e)
        error_str = str(e).lower()
        error_type = type(e).__name__

        if "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str or "401" in error_str:
            raise ConnectionError(
                "OpenAI API authentication failed. Please check your API key."
            ) from e
        elif "rate limit" in error_str or "429" in error_str:
            raise ConnectionError(
                "OpenAI API rate limit exceeded. Please wait a moment and try again."
            ) from e
        elif "timeout" in error_str or "timed out" in error_str:
            raise ConnectionError(
                "OpenAI API request timed out. Please try again."
            ) from e
        elif "connection" in error_str or "network" in error_str:
            raise ConnectionError(
                "Could not connect to OpenAI API. Please check your internet connection."
            ) from e
        else:
            raise ConnectionError(
                f"Failed to communicate with AI service ({error_type}): {e}"
            ) from e

    if not response.choices:
        return None

    msg = response.choices[0].message

    if msg.tool_calls:
        # Return all tool calls
        actions = []
        for tc in msg.tool_calls:
            try:
                parsed_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"AI returned invalid arguments for tool '{tc.function.name}'. "
                    f"Please try rephrasing your request."
                ) from e

            actions.append({
                "tool_name": tc.function.name,
                "arguments": parsed_args
            })
        logger.info("AI response: %d tool call(s): %s", len(actions),
                    [a["tool_name"] for a in actions])
        return actions

    logger.info("AI response: no tool calls returned")
    return None
