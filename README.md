# Daytrip



## Phase7: a2a-ticket-booking-agent

Adds a separately run ticket-booking service that communicates with Daytrip over A2A and supports a user approval step for proposals.
A small Streamlit day trip planner. Enter a destination, date, group size, per-person budget, and time window to get an AI-generated itinerary through Groq, supported by attraction, weather, and route data tools.

## Run locally

Requires Python 3.10 or newer.

1. Create and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env`; set `GROQ_API_KEY` and `OPENTRIPMAP_API_KEY` (get an OpenTripMap key from its API provider).
4. Start the app: `streamlit run app.py`.

## Phase 7: A2A ticket-booking demo

Daytrip and the dummy ticket-booking agent run as separate processes. Both read `GROQ_API_KEY` from the root `.env` file (or their process environment); the key is never sent over A2A. Start the booking service in one terminal, then Daytrip in another:

```sh
.venv/bin/python -m booking_service.server
```

```sh
.venv/bin/streamlit run app.py
```

The A2A endpoint defaults to `http://127.0.0.1:8001`; change `bookingServiceUrl` in `config/app.json` or set `DAYTRIP_BOOKING_URL` for Daytrip. The service host, port, model, JSON storage path, and minimum policy values live in `booking_service/config.json`. Its local JSON file is `booking_service/bookings.json` and is ignored by Git.

### A2A in this demo

A2A (Agent2Agent) is a protocol for agents to discover one another and exchange structured tasks and results. Daytrip reads the booking agent card, sends a JSON data message as a task to the agent's JSON-RPC endpoint, and reads the task result artifact. Here, the booking agent returns a confirmation, a proposal for user approval, or a failure. The agent card advertises the agent's identity, endpoint, capabilities, and skills; it is the discovery document for the services this agent exposes.

With the default local address, these URLs are useful:

- `http://127.0.0.1:8001/.well-known/agent-card.json` — A2A agent card: advertised skills, interface, and supported input/output modes.
- `http://127.0.0.1:8001/` — A2A JSON-RPC endpoint used to send tasks; this is a protocol endpoint, not a webpage for browsing services.
- `http://127.0.0.1:8001/docs` — interactive FastAPI endpoint documentation.
- `http://127.0.0.1:8001/openapi.json` — FastAPI's OpenAPI schema for the HTTP routes.
- `http://127.0.0.1:8001/redoc` — alternate view of the OpenAPI schema.

The agent card describes the booking skill and includes an example input. The OpenAPI schema describes the HTTP routes; the A2A task's JSON data fields are documented above (`request_id`, `event_name`, `event_date`, `attendee_count`, and `amount_per_person_inr`).

The booking service writes one JSON log record per processed A2A task to its process output. Each record includes the task ID, final status, confirmation ID when present, and all five defined request parameters; credentials are excluded.

Booking requests travel as an A2A task with a JSON data artifact containing `request_id`, `event_name`, `event_date`, `attendee_count`, and `amount_per_person_inr`. The result artifact has `status` set to `confirmed`, `proposal`, or `failed`. Confirmations include a dummy confirmation ID and total; proposals include the original request, suggested attendee count and amount per person, and an explanation. Daytrip asks the user to accept or decline a proposal. Acceptance sends a new A2A task; the booking service validates it again before creating any dummy booking.

Groq and OpenTripMap API keys stay server-side and are never written to logs. Open-Meteo, OpenStreetMap Nominatim, and OSRM calls do not need API keys. Do not commit `.env`.

## Configuration and invocation logs

Edit `config/app.json` to configure the app name, provider, model, generation settings, log path, logging detail, maximum tool steps, and Groq token-rate-limit handling. The default model is `openai/gpt-oss-120b`; change it to any model ID available to your Groq account. The tool loop is capped at three calls by default, and the prompt tells the model to call each tool at most once per trip. At the limit, the app disables tools and asks the model to finish with the information already gathered. For TPM rate limits, the UI shows a live countdown and retries after `rateLimitRetryDelaySeconds` (30 seconds by default), up to `maxRateLimitRetries` (one retry by default).

Every Groq request and response, including intermediate tool-call turns, is appended to `logs/invocations.jsonl`. Records include the full messages/tools sent, complete provider response, timestamp, model, status, duration, and token usage. Each external API request is also appended to `logs/api_calls.jsonl`, including tool lookups and geocoding requests; API keys are redacted. These logs contain trip details and generated itineraries, so keep them local and handle them accordingly.

Available tools are `search_attractions` (OpenTripMap), `get_weather` (Open-Meteo), and `get_travel_time` (OSRM with OpenStreetMap geocoding for place names). Attraction coordinates can be passed to the routing tool to skip geocoding calls.

Itinerary prices and local suggestions are estimates. Confirm timings and prices before you travel.
