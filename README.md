# Daytrip



## Phase5: validated-trip-plan-memory

Adds local JSON memory for successful plans and reuses a close match by destination, date, and budget.
A small Streamlit day trip planner. Enter a destination, date, group size, per-person budget, and time window to get an AI-generated itinerary through Groq, supported by attraction, weather, and route data tools.

## Run locally

Requires Python 3.10 or newer.

1. Create and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env`; set `GROQ_API_KEY` and `OPENTRIPMAP_API_KEY` (get an OpenTripMap key from its API provider).
4. Start the app: `streamlit run app.py`.

Groq and OpenTripMap API keys stay server-side and are never written to logs. Open-Meteo, OpenStreetMap Nominatim, and OSRM calls do not need API keys. Do not commit `.env`.

## Configuration and invocation logs

Edit `config/app.json` to configure the app name, provider, model, generation settings, log path, logging detail, maximum tool steps, and Groq token-rate-limit handling. The default model is `openai/gpt-oss-120b`; change it to any model ID available to your Groq account. The tool loop is capped at three calls by default, and the prompt tells the model to call each tool at most once per trip. At the limit, the app disables tools and asks the model to finish with the information already gathered. For TPM rate limits, the UI shows a live countdown and retries after `rateLimitRetryDelaySeconds` (30 seconds by default), up to `maxRateLimitRetries` (one retry by default).

Every Groq request and response, including intermediate tool-call turns, is appended to `logs/invocations.jsonl`. Records include the full messages/tools sent, complete provider response, timestamp, model, status, duration, and token usage. Each external API request is also appended to `logs/api_calls.jsonl`, including tool lookups and geocoding requests; API keys are redacted. These logs contain trip details and generated itineraries, so keep them local and handle them accordingly.

Available tools are `search_attractions` (OpenTripMap), `get_weather` (Open-Meteo), and `get_travel_time` (OSRM with OpenStreetMap geocoding for place names). Attraction coordinates can be passed to the routing tool to skip geocoding calls.

Itinerary prices and local suggestions are estimates. Confirm timings and prices before you travel.
