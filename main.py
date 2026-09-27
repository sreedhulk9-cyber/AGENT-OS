from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agents.flight_agent import FlightAgent
from core.config import Settings
from tools.amadeus_client import AmadeusAPIError, AmadeusConfigurationError, AmadeusClient


class FlightSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    origin: str = Field(description="Origin airport IATA code, for example SFO")
    destination: str = Field(description="Destination airport IATA code, for example LHR")
    departure_date: date = Field(description="Departure date in YYYY-MM-DD format")
    return_date: date | None = Field(default=None, description="Return date for a round trip")
    adults: int = Field(default=1, ge=1, le=9)
    travel_class: Literal["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"] = "ECONOMY"
    currency_code: str = Field(default="USD", min_length=3, max_length=3)
    max_results: int = Field(default=20, ge=1, le=250)

    @field_validator("origin", "destination")
    @classmethod
    def normalize_airport_code(cls, value: str) -> str:
        value = value.upper()
        if len(value) != 3 or not value.isascii() or not value.isalpha():
            raise ValueError("Must be a three-letter IATA airport code")
        return value

    @field_validator("currency_code")
    @classmethod
    def normalize_currency_code(cls, value: str) -> str:
        value = value.upper()
        if not value.isascii() or not value.isalpha():
            raise ValueError("Must be a three-letter ISO currency code")
        return value

    @model_validator(mode="after")
    def validate_dates(self) -> "FlightSearchRequest":
        if self.departure_date < date.today():
            raise ValueError("departure_date cannot be in the past")

        if self.return_date is not None and self.return_date < self.departure_date:
            raise ValueError("return_date cannot be before departure_date")

        if self.origin == self.destination:
            raise ValueError("origin and destination must be different")
        return self


class FlightOffer(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    price: dict[str, Any]
    itineraries: list[dict[str, Any]]


class FlightSearchResponse(BaseModel):
    meta: dict[str, Any] = Field(default_factory=dict)
    data: list[FlightOffer]


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Natural language flight query from user")


class AgentChatResponse(BaseModel):
    reply: str = Field(description="Agent's natural language response and recommendations")
    flights: list[dict[str, Any]] = Field(default_factory=list, description="Raw flight offers list")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="Tool calls executed by the agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_environment()
    client = AmadeusClient(settings)
    app.state.flight_agent = FlightAgent(client, settings=settings)
    yield
    client.close()


app = FastAPI(
    title="AgentOS Flight Search API",
    description="Autonomous Flight Agent API powered by OpenAI & Amadeus.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_flight_agent() -> FlightAgent:
    return app.state.flight_agent


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/agent/status", tags=["agent"], summary="Check configuration status of agent tools")
def get_agent_status(agent: Annotated[FlightAgent, Depends(get_flight_agent)]) -> dict[str, bool]:
    return {
        "openai_configured": bool(agent._openai_client and agent._openai_client.is_configured()),
        "amadeus_configured": bool(hasattr(agent._client, "is_configured") and agent._client.is_configured()),
    }


@app.post(
    "/api/v1/agent/chat",
    response_model=AgentChatResponse,
    tags=["agent"],
    summary="Chat with the autonomous Flight Agent using natural language",
)
def chat_with_agent(
    request: AgentChatRequest,
    agent: Annotated[FlightAgent, Depends(get_flight_agent)],
) -> AgentChatResponse:
    result = agent.process_query(request.message)
    return AgentChatResponse(
        reply=result.get("reply", ""),
        flights=result.get("flights", []),
        tool_calls=result.get("tool_calls", []),
    )


@app.post(
    "/api/v1/flights/search",
    response_model=FlightSearchResponse,
    tags=["flights"],
    summary="Search flight offers",
)
def search_flights(
    request: FlightSearchRequest,
    agent: Annotated[FlightAgent, Depends(get_flight_agent)],
) -> FlightSearchResponse:
    try:
        result = agent.search_flights(
            origin=request.origin,
            destination=request.destination,
            departure_date=request.departure_date.isoformat(),
            return_date=request.return_date.isoformat() if request.return_date else None,
            adults=request.adults,
            travel_class=request.travel_class,
            currency_code=request.currency_code,
            max_results=request.max_results,
        )
    except AmadeusConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except AmadeusAPIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return FlightSearchResponse(
        meta=result.get("meta", {}),
        data=result.get("data", []),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
