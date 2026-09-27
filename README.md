# AgentOS — Autonomous Multi-Agent AI System

### Milestone 1: Autonomous Flight Agent

A reference implementation of an autonomous AI agent in **AgentOS**. The Flight Agent interprets natural-language user requests using OpenAI tool/function calling, queries real-time flight data via the Amadeus Self-Service API, and provides synthesized recommendations along with structured flight offers.

---

## Architecture Flow

```
USER
 ↓
STREAMLIT FRONTEND (frontend.py)
 ↓
FASTAPI BACKEND (main.py:8000)
 ↓
FLIGHT AGENT (agents/flight_agent.py)
 ↓
OPENAI (Function Calling) + AMADEUS FLIGHT API (tools/amadeus_client.py)
 ↓
FLIGHT RESULTS
 ↓
FASTAPI
 ↓
STREAMLIT FRONTEND (Conversational Advice + Interactive Flight Cards)
```

---

## Setup & Configuration

1. **Configure Environment Variables**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Add your API keys in `.env`:
   ```ini
   AMADEUS_ENV=test
   AMADEUS_CLIENT_ID=your_amadeus_client_id
   AMADEUS_CLIENT_SECRET=your_amadeus_client_secret
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-4o-mini
   ```

2. **Install Dependencies**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

---

## Running the Application

### 1. Start the FastAPI Backend
```powershell
python main.py
```
* Backend runs on: `http://127.0.0.1:8000`
* Interactive API docs: `http://127.0.0.1:8000/docs`

### 2. Start the Streamlit Frontend
In a new terminal:
```powershell
streamlit run frontend.py
```
* Frontend opens at: `http://localhost:8501`

---

## API Endpoints

- `GET /health` — Backend health check.
- `POST /api/v1/agent/chat` — Autonomous AI Flight Agent chat endpoint:
  ```json
  {
    "message": "Find one-way flights from New York to London next Friday"
  }
  ```
- `POST /api/v1/flights/search` — Direct structured flight search endpoint.

---

## Running Tests

```powershell
pytest
```
