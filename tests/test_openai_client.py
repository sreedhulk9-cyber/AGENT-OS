import pytest

from core.config import Settings
from tools.openai_client import OpenAIClient, OpenAIConfigurationError


def test_openai_client_unconfigured():
    settings = Settings(
        client_id="id",
        client_secret="secret",
        base_url="https://test.api.amadeus.com",
        openai_api_key=None,
    )
    client = OpenAIClient(settings=settings)
    assert not client.is_configured()
    with pytest.raises(OpenAIConfigurationError, match="OpenAI API key is missing"):
        client._get_api_key()


def test_openai_client_configured():
    settings = Settings(
        client_id="id",
        client_secret="secret",
        base_url="https://test.api.amadeus.com",
        openai_api_key="sk-test-key",
    )
    client = OpenAIClient(settings=settings)
    assert client.is_configured()
    assert client._get_api_key() == "sk-test-key"


def test_openai_client_placeholder_rejected():
    client = OpenAIClient(api_key="YOUR_OPENAI_API_KEY_HERE")
    assert not client.is_configured()
    with pytest.raises(OpenAIConfigurationError, match="Please replace YOUR_OPENAI_API_KEY_HERE in .env"):
        client._get_api_key()


def test_openai_client_test_connection_unconfigured():
    client = OpenAIClient(api_key="YOUR_OPENAI_API_KEY_HERE")
    res = client.test_connection()
    assert res == {"configured": "False", "connection": "NOT TESTED"}
