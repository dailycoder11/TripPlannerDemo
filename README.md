# Daytrip



## Phase 8: request-screening-and-tool-validation

**Security focus:** this stage adds layered defenses against prompt injection and task drift in trip requests and external tool data. A prompt injection is text that tries to make the model ignore its intended task or rules—for example, instructions to change roles, reveal prompts or secrets, access other users' data, or perform an unrelated action.

### Prompt-injection screening

Before memory lookup or itinerary planning, the configurable screening model classifies the submitted request and extracts only the supported trip fields. It is instructed to treat quoted or embedded instructions as untrusted content to classify, not directions to follow. Application code validates the response and checks that extracted values still match the submitted form. If the screen flags suspicious or unrelated instructions—or its response is malformed, invalid, or unavailable—the request stops. Unscreened input is not passed to memory or the planner.

### Untrusted tool data

Prompt injection can also arrive in external data. Before results reach the planner, tool arguments are checked against each tool's allowed schema and limits. Tool results are validated: attraction descriptions are discarded, retained names and categories are screened for suspicious text, and coordinates are range-checked; weather dates and measurements are checked; route labels and distance/time values are checked. Invalid or suspicious results are withheld and stop planning. The harness also limits each tool to one execution per request.

These controls reduce risk but do not guarantee that every injection attempt will be detected. The screening model can miss attacks or flag benign requests, and text-pattern checks cannot prove arbitrary content harmless. See **Phase 8: trip-request safety screening** below for detailed behavior and configuration.

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

The booking service writes one JSON log record per processed A2A task to its process output. Each record includes the task ID, final status, confirmation ID when present, and all five defined request parameters; credentials are excluded.

Booking requests travel as an A2A task with a JSON data artifact containing `request_id`, `event_name`, `event_date`, `attendee_count`, and `amount_per_person_inr`. The result artifact has `status` set to `confirmed`, `proposal`, or `failed`. Confirmations include a dummy confirmation ID and total; proposals include the original request, suggested attendee count and amount per person, and an explanation. Daytrip asks the user to accept or decline a proposal. Acceptance sends a new A2A task; the booking service validates it again before creating any dummy booking.

Groq and OpenTripMap API keys stay server-side and are never written to logs. Open-Meteo, OpenStreetMap Nominatim, and OSRM calls do not need API keys. Do not commit `.env`.

## Phase 8: trip-request safety screening

Before searching shared reusable trip memory or calling the itinerary planner, Daytrip sends the submitted trip fields to Groq using the configurable `screeningModel` in `config/app.json` (default: `openai/gpt-oss-20b`). The screening call has no tools. It classifies suspicious or out-of-scope instructions and extracts the supported trip fields into a fixed JSON shape. Daytrip validates that response locally and rejects any unexpected changes to the submitted values.

If the screen flags the request, returns malformed data, fails validation, or is unavailable, Daytrip displays a safety message and stops that request. It does not fall back to sending unscreened input to the planner. Passing requests continue through the existing shared validated-plan memory/reuse flow, planner tool harness, deterministic itinerary evaluator, critic, and booking approval flow. The screening model is a guardrail and can still miss attacks or flag benign input.

Before external tool data reaches the planner, the tool harness validates each tool's arguments and returned structure. Attraction descriptions are removed; retained names and categories are checked for unsafe text, and coordinates are range-checked. Weather dates, codes, hourly entries, and measurements are checked; route labels must match the requested endpoints and distance/time values must be finite and nonnegative. Invalid or suspicious arguments stop the request with a safety message; invalid, suspicious, or inconsistent tool results are withheld and stop planning. The harness enforces at most one execution per tool per request.

## Configuration and invocation logs

Edit `config/app.json` to configure the app name, provider, planner model, screening model, critic model, generation settings, log path, logging detail, maximum tool steps, and Groq token-rate-limit handling. The planner defaults to `openai/gpt-oss-120b`; the screening and critic default to `openai/gpt-oss-20b`. Change model IDs to models available to your Groq account. The tool loop is capped at three successful calls by default, and the harness also enforces at most one call per tool per request. At the total limit, the app disables tools and asks the model to finish with the information already gathered. For TPM rate limits, the UI shows a live countdown and retries after `rateLimitRetryDelaySeconds` (30 seconds by default), up to `maxRateLimitRetries` (one retry by default).

Every Groq request and response, including intermediate tool-call turns, is appended to `logs/invocations.jsonl`. Records include the full messages/tools sent, complete provider response, timestamp, model, status, duration, and token usage. Each external API request is also appended to `logs/api_calls.jsonl`, including tool lookups and geocoding requests; API keys are redacted. These logs contain trip details and generated itineraries, so keep them local and handle them accordingly.

Available tools are `search_attractions` (OpenTripMap), `get_weather` (Open-Meteo), and `get_travel_time` (OSRM with OpenStreetMap geocoding for place names). Attraction coordinates can be passed to the routing tool to skip geocoding calls.

Itinerary prices and local suggestions are estimates. Confirm timings and prices before you travel.
