import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

# Configure Streamlit page
st.set_page_config(
    page_title="AgentOS - Flight Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://127.0.0.1:8000"


def check_backend_health() -> bool:
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return res.status_code == 200 and res.json().get("status") == "ok"
    except Exception:
        return False


def check_agent_status() -> dict[str, bool]:
    try:
        res = requests.get(f"{BACKEND_URL}/api/v1/agent/status", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"openai_configured": False, "amadeus_configured": False}


def send_chat_query(message: str) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/agent/chat",
            json={"message": message},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        return {
            "reply": f"Backend returned HTTP {response.status_code}: {response.text}",
            "flights": [],
            "tool_calls": [],
        }
    except requests.exceptions.ConnectionError:
        return {
            "reply": (
                "⚠️ Could not connect to the FastAPI backend at http://127.0.0.1:8000. "
                "Please make sure the backend is running (`python main.py`)."
            ),
            "flights": [],
            "tool_calls": [],
        }
    except Exception as e:
        return {"reply": f"Error communicating with backend: {str(e)}", "flights": [], "tool_calls": []}


def format_duration(iso_duration: str) -> str:
    """Format ISO 8601 duration (e.g. PT10H30M) into human readable format."""
    if not iso_duration or not iso_duration.startswith("PT"):
        return iso_duration
    clean = iso_duration.replace("PT", "")
    hours = ""
    minutes = ""
    if "H" in clean:
        parts = clean.split("H")
        hours = f"{parts[0]}h "
        clean = parts[1] if len(parts) > 1 else ""
    if "M" in clean:
        minutes = clean.replace("M", "m")
    return f"{hours}{minutes}".strip() or iso_duration


# Sidebar
with st.sidebar:
    st.title("🤖 AgentOS")
    st.markdown("**Autonomous Multi-Agent AI System**")
    st.markdown("---")

    # Backend status badge
    is_online = check_backend_health()
    if is_online:
        st.success("🟢 FastAPI Backend: Online")
    else:
        st.error("🔴 FastAPI Backend: Offline")
        st.caption("Start it with `python main.py` in your terminal.")

    # API Configuration status
    agent_status = check_agent_status()
    st.markdown("### 🔑 Credentials Status")
    if agent_status.get("openai_configured"):
        st.success("OpenAI Key: Configured")
    else:
        st.warning("OpenAI Key: Missing in .env")

    if agent_status.get("amadeus_configured"):
        st.success("Amadeus API: Configured")
    else:
        st.warning("Amadeus API: Missing in .env")

    st.markdown("---")
    st.subheader("💡 Try Example Queries")
    example_prompts = [
        "Find flights from Kochi to Delhi.",
        "Find one-way flights from New York to London next Friday.",
        "Round trip from SFO to Tokyo next month for 2 passengers.",
        "Cheapest flights from Paris to Dubai departing 2026-11-15.",
    ]
    for prompt in example_prompts:
        if st.button(prompt, use_container_width=True):
            st.session_state["user_prompt"] = prompt

    st.markdown("---")
    st.caption("Milestone 1: Flight Agent (OpenAI + Amadeus API)")

# Main Header
st.title("✈️ Autonomous Flight Agent")
st.markdown(
    "Ask any flight request in plain English. The AI Flight Agent extracts your travel intent, "
    "calls the live Amadeus Flight Search API, and analyzes the best options for you."
)

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your AI Flight Agent. Where would you like to fly? "
                "You can tell me your origin, destination, preferred dates, and any preferences like cabin class or passengers."
            ),
            "flights": [],
            "tool_calls": [],
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Display tool call trace if available
        if msg.get("tool_calls"):
            with st.expander("🛠️ View Agent Tool Invocations"):
                for tc in msg["tool_calls"]:
                    st.code(
                        f"Function: {tc.get('function')}\nArguments: {json.dumps(tc.get('arguments', {}), indent=2)}",
                        language="json",
                    )

        # Display structured flight cards if available
        flights = msg.get("flights", [])
        if flights:
            st.markdown(f"### 📋 Found {len(flights)} Flight Offers")
            for idx, flight in enumerate(flights, start=1):
                price_info = flight.get("price", {})
                currency = price_info.get("currency", "USD")
                total_price = price_info.get("total", "N/A")
                itineraries = flight.get("itineraries", [])

                with st.container(border=True):
                    col1, col2, col3 = st.columns([3, 2, 2])
                    with col1:
                        st.subheader(f"Option {idx}: {currency} {total_price}")

                    with col2:
                        seats = flight.get("numberOfBookableSeats", "N/A")
                        st.caption(f"💺 Bookable Seats: {seats}")

                    with col3:
                        offer_id = flight.get("id", str(idx))
                        st.caption(f"ID: #{offer_id}")

                    # Render itinerary segments
                    for itin_idx, itin in enumerate(itineraries, start=1):
                        duration = format_duration(itin.get("duration", ""))
                        segments = itin.get("segments", [])
                        st.markdown(f"**Leg {itin_idx}** • Total Duration: `{duration}` • Stops: `{len(segments) - 1}`")

                        for seg in segments:
                            dep = seg.get("departure", {})
                            arr = seg.get("arrival", {})
                            carrier = seg.get("carrierCode", "")
                            flight_num = seg.get("number", "")
                            dep_time = dep.get("at", "").replace("T", " ")
                            arr_time = arr.get("at", "").replace("T", " ")

                            st.write(
                                f"✈️ **{carrier} {flight_num}**: "
                                f"`{dep.get('iataCode')}` ({dep_time}) ➔ "
                                f"`{arr.get('iataCode')}` ({arr_time})"
                            )

# User input handling
prompt_input = st.chat_input("Ask for flights (e.g., 'Find flights from JFK to LHR on 2026-10-15')")

# Check if an example prompt was clicked
if "user_prompt" in st.session_state and st.session_state["user_prompt"]:
    prompt_input = st.session_state.pop("user_prompt")

if prompt_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.write(prompt_input)

    # Call the Agent via FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Flight Agent is analyzing your request and searching live offers..."):
            result = send_chat_query(prompt_input)
            reply_text = result.get("reply", "No response received.")
            flights_data = result.get("flights", [])
            tool_calls_data = result.get("tool_calls", [])

            st.write(reply_text)

            if tool_calls_data:
                with st.expander("🛠️ View Agent Tool Invocations"):
                    for tc in tool_calls_data:
                        st.code(
                            f"Function: {tc.get('function')}\nArguments: {json.dumps(tc.get('arguments', {}), indent=2)}",
                            language="json",
                        )

            if flights_data:
                st.markdown(f"### 📋 Found {len(flights_data)} Flight Offers")
                for idx, flight in enumerate(flights_data, start=1):
                    price_info = flight.get("price", {})
                    currency = price_info.get("currency", "USD")
                    total_price = price_info.get("total", "N/A")
                    itineraries = flight.get("itineraries", [])

                    with st.container(border=True):
                        col1, col2, col3 = st.columns([3, 2, 2])
                        with col1:
                            st.subheader(f"Option {idx}: {currency} {total_price}")

                        with col2:
                            seats = flight.get("numberOfBookableSeats", "N/A")
                            st.caption(f"💺 Bookable Seats: {seats}")

                        with col3:
                            offer_id = flight.get("id", str(idx))
                            st.caption(f"ID: #{offer_id}")

                        for itin_idx, itin in enumerate(itineraries, start=1):
                            duration = format_duration(itin.get("duration", ""))
                            segments = itin.get("segments", [])
                            st.markdown(f"**Leg {itin_idx}** • Total Duration: `{duration}` • Stops: `{len(segments) - 1}`")

                            for seg in segments:
                                dep = seg.get("departure", {})
                                arr = seg.get("arrival", {})
                                carrier = seg.get("carrierCode", "")
                                flight_num = seg.get("number", "")
                                dep_time = dep.get("at", "").replace("T", " ")
                                arr_time = arr.get("at", "").replace("T", " ")

                                st.write(
                                    f"✈️ **{carrier} {flight_num}**: "
                                    f"`{dep.get('iataCode')}` ({dep_time}) ➔ "
                                    f"`{arr.get('iataCode')}` ({arr_time})"
                                )

    # Save assistant message into state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply_text,
            "flights": flights_data,
            "tool_calls": tool_calls_data,
        }
    )
