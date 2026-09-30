"""LLM client for ReviewMind.

Talks to any OpenAI-compatible chat-completions API. The provider, model,
base URL and timeouts are configurable through ``config.yaml`` and
environment variables, so the LLM provider can be swapped without touching
application logic.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env once at import time so config + API key are available everywhere.
load_dotenv(PROJECT_ROOT / ".env")
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class LLMError(Exception):
    """Base error for LLM communication failures."""


class LLMConfigError(LLMError):
    """Raised when configuration or API key is missing/invalid."""


class LLMAPIError(LLMError):
    """Raised when the LLM API call fails (network, timeout, rate limit)."""


class LLMResponseError(LLMError):
    """Raised when the API responds but the payload cannot be parsed."""


def load_config() -> dict:
    """Load config.yaml, with environment variable overrides."""
    config: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}

    llm = config.setdefault("llm", {})
    llm["provider"] = os.getenv("LLM_PROVIDER", llm.get("provider", "openrouter"))
    llm["model"] = os.getenv("LLM_MODEL", llm.get("model", ""))
    llm["base_url"] = os.getenv("LLM_BASE_URL", llm.get("base_url", ""))
    llm["timeout"] = int(os.getenv("LLM_TIMEOUT", str(llm.get("timeout", 60))))
    llm["max_retries"] = int(os.getenv("LLM_MAX_RETRIES", str(llm.get("max_retries", 2))))
    llm["temperature"] = float(os.getenv("LLM_TEMPERATURE", str(llm.get("temperature", 0.2))))
    llm["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", str(llm.get("max_tokens", 1500))))
    llm["max_reasoning_tokens"] = int(
        os.getenv("LLM_MAX_REASONING_TOKENS", str(llm.get("max_reasoning_tokens", 0)))
    )

    config.setdefault("app", {})["max_input_chars"] = int(
        os.getenv("MAX_INPUT_CHARS", str(config.get("app", {}).get("max_input_chars", 20000)))
    )
    return config


def get_api_key(config: dict) -> str:
    """Resolve the API key from the environment."""
    key = os.getenv("LLM_API_KEY", "").strip()
    if not key:
        raise LLMConfigError(
            "Missing API key. Set LLM_API_KEY in your .env file (see .env.example)."
        )
    return key


def _error_detail(response) -> str:
    """Extract the provider's own error message, if the body is JSON."""
    try:
        body = response.json()
    except ValueError:
        return ""
    message = body.get("error", {}).get("message") if isinstance(body, dict) else None
    return f" {message}" if message else ""


class LLMClient:
    """Minimal client for OpenAI-compatible chat-completions endpoints."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        llm = self.config["llm"]
        self.model = llm["model"]
        self.base_url = llm["base_url"].rstrip("/")
        self.timeout = llm["timeout"]
        self.max_retries = llm["max_retries"]
        self.temperature = llm["temperature"]
        self.max_tokens = llm["max_tokens"]
        self.max_reasoning_tokens = llm.get("max_reasoning_tokens", 0)

    # ------------------------------------------------------------------
    def chat(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt to the LLM and return the assistant message text."""
        if not self.model or not self.base_url:
            raise LLMConfigError(
                "LLM model/base URL not configured. Check config/config.yaml and .env"
            )

        api_key = get_api_key(self.config)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.max_reasoning_tokens:
            payload["reasoning"] = {"max_tokens": self.max_reasoning_tokens}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                if response.status_code == 401:
                    raise LLMConfigError("API key rejected (HTTP 401). Check LLM_API_KEY.")
                if response.status_code == 429:
                    last_error = LLMAPIError("Rate limit reached (HTTP 429).")
                    if attempt < self.max_retries:
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise last_error
                if response.status_code >= 400:
                    raise LLMAPIError(
                        f"LLM API error (HTTP {response.status_code}).{_error_detail(response)}"
                    )

                data = response.json()
                try:
                    choice = data["choices"][0]
                    content = choice["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise LLMResponseError("Unexpected response shape from LLM API.") from exc
                if not content or not content.strip():
                    reasoning = choice["message"].get("reasoning") or ""
                    if choice.get("finish_reason") == "length" or reasoning:
                        raise LLMResponseError(
                            "LLM returned no content: the token budget was consumed by "
                            f"reasoning. Raise LLM_MAX_TOKENS (currently {self.max_tokens}) "
                            "or cap reasoning via LLM_MAX_REASONING_TOKENS."
                        )
                    raise LLMResponseError("LLM returned an empty response.")
                return content

            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 * (attempt + 1))
            except (LLMConfigError, LLMAPIError, LLMResponseError):
                raise

        raise LLMAPIError(
            f"Could not reach the LLM API after {self.max_retries + 1} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def extract_json(text: str) -> Any:
        """Parse the first JSON object found in an LLM response."""
        cleaned = text.strip()
        # Strip markdown code fences if present.
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Fallback: grab text between the first { and last }.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise LLMResponseError("LLM returned invalid JSON.") from exc
        raise LLMResponseError("No JSON object found in LLM response.")
