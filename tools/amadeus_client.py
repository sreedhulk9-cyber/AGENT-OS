from os import getenv
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

from dotenv import load_dotenv
import httpx

from core.config import Settings

DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def is_real_amadeus_val(val: str | None) -> bool:
    if not val:
        return False
    v = val.strip()
    if not v:
        return False
    if v.startswith("YOUR_") or v.startswith("your_") or "YOUR_AMADEUS_" in v:
        return False
    return True


class AmadeusConfigurationError(Exception):
    pass


class AmadeusAPIError(Exception):
    pass


class AmadeusClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._http = http_client or httpx.Client(
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
        )
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = Lock()

    def close(self) -> None:
        self._http.close()

    def is_configured(self) -> bool:
        if self._settings and is_real_amadeus_val(self._settings.client_id) and is_real_amadeus_val(self._settings.client_secret):
            return True
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)
        cid = getenv("AMADEUS_CLIENT_ID")
        sec = getenv("AMADEUS_CLIENT_SECRET")
        return is_real_amadeus_val(cid) and is_real_amadeus_val(sec)

    def search_flights(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: str | None,
        adults: int,
        travel_class: str,
        currency_code: str,
        max_results: int,
    ) -> dict[str, Any]:
        params: dict[str, str | int] = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date,
            "adults": adults,
            "travelClass": travel_class,
            "currencyCode": currency_code,
            "max": max_results,
        }
        if return_date:
            params["returnDate"] = return_date

        try:
            response = self._http.get(
                "/v2/shopping/flight-offers",
                params=params,
                headers={"Authorization": f"Bearer {self._get_access_token()}"},
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise AmadeusAPIError("Amadeus flight search timed out.") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise AmadeusAPIError("Amadeus flight search rate limit was reached.") from exc
            raise AmadeusAPIError("Amadeus could not complete the flight search.") from exc
        except httpx.RequestError as exc:
            raise AmadeusAPIError("Could not connect to Amadeus flight search.") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise AmadeusAPIError("Amadeus returned an invalid flight-search response.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data", []), list):
            raise AmadeusAPIError("Amadeus returned an invalid flight-search response.")
        return payload

    def _get_access_token(self) -> str:
        with self._token_lock:
            if self._token is not None and monotonic() < self._token_expires_at:
                return self._token

            client_id = self._settings.client_id if self._settings and is_real_amadeus_val(self._settings.client_id) else None
            client_secret = self._settings.client_secret if self._settings and is_real_amadeus_val(self._settings.client_secret) else None

            if not client_id or not client_secret:
                load_dotenv(dotenv_path=DOTENV_PATH, override=True)
                client_id = getenv("AMADEUS_CLIENT_ID")
                client_secret = getenv("AMADEUS_CLIENT_SECRET")

            if not is_real_amadeus_val(client_id) or not is_real_amadeus_val(client_secret):
                raise AmadeusConfigurationError(
                    "Flight API credentials are not configured (credentials are missing). "
                    "Set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET in the .env file."
                )

            try:
                response = self._http.post(
                    "/v1/security/oauth2/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": client_id,
                        "client_secret": client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.TimeoutException as exc:
                raise AmadeusAPIError("Amadeus authentication timed out.") from exc
            except httpx.HTTPStatusError as exc:
                raise AmadeusAPIError("Amadeus rejected the configured credentials.") from exc
            except httpx.RequestError as exc:
                raise AmadeusAPIError("Could not connect to Amadeus authentication.") from exc
            except ValueError as exc:
                raise AmadeusAPIError("Amadeus returned an invalid authentication response.") from exc

            token = payload.get("access_token") if isinstance(payload, dict) else None
            expires_in = payload.get("expires_in", 0) if isinstance(payload, dict) else 0
            if not isinstance(token, str) or not token:
                raise AmadeusAPIError("Amadeus authentication response did not include a token.")
            if not isinstance(expires_in, (int, float)) or expires_in <= 0:
                raise AmadeusAPIError("Amadeus authentication response had an invalid expiry.")

            self._token = token
            self._token_expires_at = monotonic() + max(0, expires_in - 60)
            return token
