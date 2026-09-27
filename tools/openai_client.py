import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from core.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


class OpenAIConfigurationError(Exception):
    """Raised when OpenAI configuration or API key is missing."""
    pass


class OpenAIAPIError(Exception):
    """Raised when an OpenAI API request fails."""
    pass


def is_real_openai_key(key: str | None) -> bool:
    if not key:
        return False
    k = key.strip()
    if not k:
        return False
    if k.startswith("YOUR_") or k.startswith("your_") or "YOUR_OPENAI_API_KEY_HERE" in k:
        return False
    return True


class OpenAIClient:
    """Client for interacting with the OpenAI API for LLM and tool calling."""

    def __init__(
        self,
        settings: Settings | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._settings = settings
        self._api_key = api_key
        self._model = model or (settings.openai_model if settings else None)

    def _get_api_key(self) -> str:
        """Safely retrieve the OpenAI API key from explicit arg, settings, or .env."""
        if self._api_key:
            if is_real_openai_key(self._api_key):
                return self._api_key
            raise OpenAIConfigurationError(
                "OpenAI API key is missing. Please replace YOUR_OPENAI_API_KEY_HERE in .env with your real OpenAI API key, then save the file."
            )

        if self._settings and self._settings.openai_api_key:
            k = self._settings.openai_api_key.strip()
            if is_real_openai_key(k):
                return k

        # Check project root .env
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)
        key_from_env = os.getenv("OPENAI_API_KEY")
        if key_from_env:
            k = key_from_env.strip()
            if is_real_openai_key(k):
                return k
            if "YOUR_OPENAI_API_KEY_HERE" in k or k.startswith("YOUR_") or k.startswith("your_"):
                raise OpenAIConfigurationError(
                    "OpenAI API key is missing. Please replace YOUR_OPENAI_API_KEY_HERE in .env with your real OpenAI API key, then save the file."
                )

        raise OpenAIConfigurationError(
            "OpenAI API key is missing. Please set OPENAI_API_KEY in your .env file "
            "so the Flight Agent can process your natural language request."
        )

    def _get_model(self) -> str:
        """Retrieve the configured model name."""
        if self._model:
            return self._model
        if self._settings and self._settings.openai_model:
            return self._settings.openai_model
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def is_configured(self) -> bool:
        """Check if an API key is currently available without throwing an exception."""
        try:
            key = self._get_api_key()
            return bool(key and is_real_openai_key(key))
        except OpenAIConfigurationError:
            return False

    def test_connection(self) -> dict[str, str]:
        """Test connection to OpenAI without leaking credentials."""
        if not self.is_configured():
            return {"configured": "False", "connection": "NOT TESTED"}
        try:
            client = OpenAI(api_key=self._get_api_key())
            client.models.list()
            return {"configured": "True", "connection": "SUCCESS"}
        except AuthenticationError:
            return {"configured": "True", "connection": "FAILED (AuthenticationError: Invalid API key)"}
        except Exception as exc:
            return {"configured": "True", "connection": f"FAILED ({str(exc)})"}

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
    ) -> Any:
        """Call OpenAI chat completions with error handling and optional tool calling."""
        api_key = self._get_api_key()
        model = self._get_model()

        client = OpenAI(api_key=api_key)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        try:
            return client.chat.completions.create(**kwargs)
        except AuthenticationError as exc:
            raise OpenAIAPIError(
                "OpenAI rejected the configured API key. Please check that OPENAI_API_KEY "
                "in your .env file is valid."
            ) from exc
        except RateLimitError as exc:
            raise OpenAIAPIError(
                "OpenAI rate limit or quota exceeded. Please check your OpenAI account billing."
            ) from exc
        except APIConnectionError as exc:
            raise OpenAIAPIError(
                "Could not connect to OpenAI API. Please check your internet connection."
            ) from exc
        except OpenAIError as exc:
            raise OpenAIAPIError(f"OpenAI service error: {str(exc)}") from exc
        except Exception as exc:
            raise OpenAIAPIError(f"Unexpected error while communicating with OpenAI: {str(exc)}") from exc
