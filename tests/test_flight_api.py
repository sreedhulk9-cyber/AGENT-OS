from datetime import date, timedelta

from fastapi.testclient import TestClient

import main


class StubFlightAgent:
    def search_flights(self, **kwargs):
        assert kwargs["origin"] == "SFO"
        assert kwargs["destination"] == "LHR"
        return {
            "meta": {"count": 1},
            "data": [
                {
                    "id": "1",
                    "price": {"currency": "USD", "total": "500.00"},
                    "itineraries": [{"duration": "PT10H"}],
                }
            ],
        }

    def process_query(self, message: str):
        return {
            "reply": f"Found flights for: {message}",
            "flights": [{"id": "1", "price": {"currency": "USD", "total": "500.00"}}],
            "tool_calls": [{"function": "search_flight_offers", "arguments": {"origin": "SFO", "destination": "LHR"}}],
        }


def test_health_endpoint():
    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_flight_search_normalizes_codes_and_returns_offers():
    main.app.dependency_overrides[main.get_flight_agent] = StubFlightAgent
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/v1/flights/search",
                json={
                    "origin": "sfo",
                    "destination": "lhr",
                    "departure_date": str(date.today() + timedelta(days=30)),
                },
            )
        assert response.status_code == 200
        assert response.json()["meta"] == {"count": 1}
        assert response.json()["data"][0]["price"]["total"] == "500.00"
    finally:
        main.app.dependency_overrides.pop(main.get_flight_agent, None)


def test_flight_search_rejects_invalid_airport_code():
    with TestClient(main.app) as client:
        response = client.post(
            "/api/v1/flights/search",
            json={
                "origin": "SFO1",
                "destination": "LHR",
                "departure_date": str(date.today() + timedelta(days=30)),
            },
        )
    assert response.status_code == 422


def test_flight_search_rejects_return_before_departure():
    with TestClient(main.app) as client:
        response = client.post(
            "/api/v1/flights/search",
            json={
                "origin": "SFO",
                "destination": "LHR",
                "departure_date": str(date.today() + timedelta(days=30)),
                "return_date": str(date.today() + timedelta(days=29)),
            },
        )
    assert response.status_code == 422


def test_agent_chat_endpoint():
    main.app.dependency_overrides[main.get_flight_agent] = StubFlightAgent
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/v1/agent/chat",
                json={"message": "Flights to London"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "Found flights for: Flights to London" in data["reply"]
        assert len(data["flights"]) == 1
        assert len(data["tool_calls"]) == 1
    finally:
        main.app.dependency_overrides.pop(main.get_flight_agent, None)


def test_agent_status_endpoint():
    with TestClient(main.app) as client:
        response = client.get("/api/v1/agent/status")
        assert response.status_code == 200
        assert "openai_configured" in response.json()
        assert "amadeus_configured" in response.json()
