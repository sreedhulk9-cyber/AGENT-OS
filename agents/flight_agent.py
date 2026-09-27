import json
from datetime import date
from typing import Any

from core.config import Settings
from tools.amadeus_client import AmadeusAPIError, AmadeusConfigurationError, AmadeusClient
from tools.openai_client import OpenAIAPIError, OpenAIConfigurationError, OpenAIClient

FLIGHT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flight_offers",
            "description": (
                "Search live flight offers between two airports on specified dates using the Amadeus Flight API. "
                "Origin and destination MUST be standard 3-letter IATA airport codes (e.g., SFO, JFK, LHR, CDG, DXB, COK, DEL). "
                "Departure date must be in YYYY-MM-DD format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "3-letter IATA code of origin airport (e.g., 'COK', 'JFK', 'SFO', 'DEL')",
                    },
                    "destination": {
                        "type": "string",
                        "description": "3-letter IATA code of destination airport (e.g., 'DEL', 'LHR', 'CDG', 'BOM')",
                    },
                    "departure_date": {
                        "type": "string",
                        "description": "Departure date in ISO format (YYYY-MM-DD)",
                    },
                    "return_date": {
                        "type": "string",
                        "description": "Optional return date in ISO format (YYYY-MM-DD) for round trips",
                    },
                    "adults": {
                        "type": "integer",
                        "description": "Number of adult travelers (1 to 9). Default is 1.",
                        "default": 1,
                    },
                    "travel_class": {
                        "type": "string",
                        "enum": ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"],
                        "description": "Cabin travel class. Default is ECONOMY.",
                        "default": "ECONOMY",
                    },
                    "currency_code": {
                        "type": "string",
                        "description": "3-letter ISO currency code (e.g., 'INR', 'USD', 'EUR'). Default is 'USD' or local currency.",
                        "default": "USD",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of flight offers to retrieve (1 to 20). Default is 10.",
                        "default": 10,
                    },
                },
                "required": ["origin", "destination", "departure_date"],
            },
        },
    }
]


class FlightAgent:
    """Autonomous AI Flight Agent for AgentOS.

    Interprets natural language queries, resolves travel parameters,
    executes real flight searches via Amadeus API, and synthesizes results.
    """

    def __init__(
        self,
        client: AmadeusClient,
        settings: Settings | None = None,
        openai_client: OpenAIClient | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._openai_client = openai_client or OpenAIClient(settings=settings)

    def search_flights(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: str | None = None,
        adults: int = 1,
        travel_class: str = "ECONOMY",
        currency_code: str = "USD",
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Direct search delegator preserving backwards compatibility with existing endpoints."""
        return self._client.search_flights(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            adults=adults,
            travel_class=travel_class,
            currency_code=currency_code,
            max_results=max_results,
        )

    def process_query(self, user_query: str) -> dict[str, Any]:
        """Process a natural language user request using OpenAI tool calling and Amadeus.

        Returns:
            dict containing:
                - reply (str): Natural language explanation / recommendation from LLM.
                - flights (list): List of live flight offers retrieved from Amadeus.
                - tool_calls (list): Tool invocation details, if any.
        """
        # Step 1: Verify OpenAI configuration
        if not self._openai_client.is_configured():
            try:
                self._openai_client._get_api_key()
            except OpenAIConfigurationError as exc:
                return {
                    "reply": str(exc),
                    "flights": [],
                    "tool_calls": [],
                }
            return {
                "reply": (
                    "OpenAI API key is missing. Please set OPENAI_API_KEY in your .env file "
                    "so the Flight Agent can process your natural language request."
                ),
                "flights": [],
                "tool_calls": [],
            }

        today_str = date.today().isoformat()

        system_prompt = (
            f"You are the Flight Agent in AgentOS, an autonomous multi-agent AI system.\n"
            f"Today's date is: {today_str}.\n\n"
            f"Your responsibilities:\n"
            f"1. Understand natural-language user travel queries (origin, destination, dates, cabin class, passengers).\n"
            f"2. Resolve city/airport names to standard 3-letter IATA airport codes:\n"
            f"   - Examples: Kochi -> COK, Delhi -> DEL, Mumbai -> BOM, Bengaluru -> BLR, Chennai -> MAA, "
            f"Hyderabad -> HYD, New York -> JFK or EWR, London -> LHR, Paris -> CDG, Tokyo -> HND or NRT, "
            f"San Francisco -> SFO, Dubai -> DXB, Singapore -> SIN.\n"
            f"3. Call the 'search_flight_offers' tool with the exact extracted parameters.\n"
            f"4. If no departure date is specified by the user, assume a date in the near future (e.g. 7 days from today: {today_str}).\n"
            f"5. If essential parameters (like origin or destination) are completely missing, politely ask the user for details.\n"
            f"6. When the tool returns flight offers, summarize the best options clearly:\n"
            f"   - Highlight the lowest price, fastest flight, airline names, and any layovers.\n"
            f"   - Keep your response helpful, concise, and structured.\n"
            f"7. If the flight API returns an error or says credentials are not configured, explain that clearly to the user "
            f"and do not fabricate fake flight results.\n"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        try:
            response = self._openai_client.create_chat_completion(
                messages=messages,
                tools=FLIGHT_TOOLS,
                tool_choice="auto",
            )
        except OpenAIConfigurationError as exc:
            return {
                "reply": str(exc),
                "flights": [],
                "tool_calls": [],
            }
        except OpenAIAPIError as exc:
            return {
                "reply": str(exc),
                "flights": [],
                "tool_calls": [],
            }
        except Exception as exc:
            return {
                "reply": f"OpenAI error: {str(exc)}",
                "flights": [],
                "tool_calls": [],
            }

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls or []

        # If LLM did not call any tool, return its direct text response
        if not tool_calls:
            return {
                "reply": response_message.content or "How can I assist you with your flight search?",
                "flights": [],
                "tool_calls": [],
            }

        # Handle tool calls
        retrieved_flights: list[dict[str, Any]] = []
        executed_tool_details: list[dict[str, Any]] = []

        assistant_msg_dict: dict[str, Any] = {
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg_dict)

        for tc in tool_calls:
            if tc.function.name == "search_flight_offers":
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                executed_tool_details.append({"function": tc.function.name, "arguments": args})

                # Check if Amadeus credentials are configured
                origin_code = args.get("origin", "").upper()
                dest_code = args.get("destination", "").upper()
                dep_date = args.get("departure_date", "")

                try:
                    search_result = self.search_flights(
                        origin=origin_code,
                        destination=dest_code,
                        departure_date=dep_date,
                        return_date=args.get("return_date"),
                        adults=int(args.get("adults", 1)),
                        travel_class=args.get("travel_class", "ECONOMY"),
                        currency_code=args.get("currency_code", "USD"),
                        max_results=int(args.get("max_results", 10)),
                    )
                    flight_offers = search_result.get("data", [])
                    retrieved_flights.extend(flight_offers)
                    tool_content = json.dumps({"count": len(flight_offers), "data": flight_offers[:5]})
                except AmadeusConfigurationError:
                    tool_content = json.dumps(
                        {"error": "Flight API credentials are not configured. Please set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET in your .env file."}
                    )
                except AmadeusAPIError as exc:
                    tool_content = json.dumps({"error": f"Flight API error: {str(exc)}"})
                except Exception as exc:
                    tool_content = json.dumps({"error": f"Flight search failed: {str(exc)}"})

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": tool_content,
                    }
                )

        # Get final conversational synthesis from OpenAI
        try:
            second_response = self._openai_client.create_chat_completion(
                messages=messages,
            )
            final_reply = second_response.choices[0].message.content or ""
        except OpenAIAPIError as exc:
            final_reply = (
                f"Retrieved flight search status, but encountered an error generating summary: {str(exc)}"
            )
        except Exception as exc:
            final_reply = (
                f"Retrieved {len(retrieved_flights)} flight offers, "
                f"but encountered an error generating summary: {str(exc)}"
            )

        return {
            "reply": final_reply,
            "flights": retrieved_flights,
            "tool_calls": executed_tool_details,
        }
