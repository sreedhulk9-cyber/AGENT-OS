import httpx
import pytest

from core.config import Settings
from tools.amadeus_client import AmadeusClient, AmadeusConfigurationError


def test_search_authenticates_caches_token_and_sends_search_parameters():
    requests = []

    def handle_request(request):
        requests.append(request)
        if request.url.path.endswith("/oauth2/token"):
            return httpx.Response(
                200,
                json={"access_token": "test-token", "expires_in": 1800},
            )
        return httpx.Response(200, json={"meta": {"count": 0}, "data": []})

    client = AmadeusClient(
        Settings("client-id", "client-secret", "https://test.api.amadeus.com"),
        httpx.Client(
            base_url="https://test.api.amadeus.com",
            transport=httpx.MockTransport(handle_request),
        ),
    )
    try:
        result = client.search_flights(
            origin="SFO",
            destination="LHR",
            departure_date="2027-06-15",
            return_date="2027-06-22",
            adults=2,
            travel_class="ECONOMY",
            currency_code="USD",
            max_results=10,
        )
        client.search_flights(
            origin="SFO",
            destination="LHR",
            departure_date="2027-06-15",
            return_date=None,
            adults=1,
            travel_class="BUSINESS",
            currency_code="EUR",
            max_results=5,
        )
    finally:
        client.close()

    assert result == {"meta": {"count": 0}, "data": []}
    assert sum(request.url.path.endswith("/oauth2/token") for request in requests) == 1
    searches = [request for request in requests if request.url.path.endswith("flight-offers")]
    assert searches[0].headers["Authorization"] == "Bearer test-token"
    assert dict(searches[0].url.params) == {
        "originLocationCode": "SFO",
        "destinationLocationCode": "LHR",
        "departureDate": "2027-06-15",
        "adults": "2",
        "travelClass": "ECONOMY",
        "currencyCode": "USD",
        "max": "10",
        "returnDate": "2027-06-22",
    }
    assert "returnDate" not in dict(searches[1].url.params)


def test_search_reports_missing_credentials():
    client = AmadeusClient(
        Settings(None, None, "https://test.api.amadeus.com"),
        httpx.Client(
            base_url="https://test.api.amadeus.com",
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        ),
    )
    try:
        with pytest.raises(AmadeusConfigurationError, match="credentials are missing"):
            client.search_flights(
                origin="SFO",
                destination="LHR",
                departure_date="2027-06-15",
                return_date=None,
                adults=1,
                travel_class="ECONOMY",
                currency_code="USD",
                max_results=10,
            )
    finally:
        client.close()
