from dataclasses import dataclass
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=True)


@dataclass(frozen=True)
class Settings:
    client_id: str | None
    client_secret: str | None
    base_url: str
    timeout_seconds: float = 20.0
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)
        environment = getenv("AMADEUS_ENV", "test").strip().lower()
        base_urls = {
            "test": "https://test.api.amadeus.com",
            "production": "https://api.amadeus.com",
        }
        if environment not in base_urls:
            raise ValueError("AMADEUS_ENV must be either 'test' or 'production'")

        return cls(
            client_id=getenv("AMADEUS_CLIENT_ID"),
            client_secret=getenv("AMADEUS_CLIENT_SECRET"),
            base_url=base_urls[environment],
            openai_api_key=getenv("OPENAI_API_KEY"),
            openai_model=getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
