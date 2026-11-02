# Daytrip

A Streamlit day trip planner. Enter a destination, date, group size, per-person budget, and time window to get an AI-generated itinerary through Groq, supported by attraction, weather, and route data tools.

## Phase 6: critic-assisted planning

Phase 6 adds a separate, advisory critic after the planner produces a structured itinerary and the deterministic evaluator returns `PASS`. The critic receives the trip requirements, final validated plan, and successful tool results. It returns a short overall assessment and up to three practical suggestions, considering schedule pacing, travel burden, meal timing, activity variety, weather, and group suitability.

The critic does not call tools, rewrite the itinerary, or change the evaluator's result. It is invoked once for a newly generated, parseable `PASS` plan. Plans that fail evaluation, cannot be parsed, or are reused directly from memory do not receive a critic call. If the critic request fails or returns invalid JSON, the validated itinerary remains available and the UI reports that critic feedback is unavailable.

The planner and critic use separate model settings. `criticModel` defaults to `openai/gpt-oss-20b`; optional `criticMaxTokens` and `criticTemperature` settings default to 600 and 0.2. The critic call follows planner calls in the request's invocation log and shares their invocation ID. Its record identifies the model, call index, request, response, status, duration, and generation settings. The per-request LLM call limit also applies to the critic; when the planner has used the full allowance, the critic is skipped.

Phase 6 retains the earlier deterministic evaluation and revision loop, file-based plan memory, and tool-assisted itinerary generation. A passing plan may be saved to memory regardless of the critic's suggestions; memory reuse continues to skip planning and critic calls.

## Run locally

Requires Python 3.10 or newer.

1. Create and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env`; set `GROQ_API_KEY` and `OPENTRIPMAP_API_KEY` (get an OpenTripMap key from its API provider).
4. Start the app: `streamlit run app.py`.

Groq and OpenTripMap API keys stay server-side and are never written to logs. Open-Meteo, OpenStreetMap Nominatim, and OSRM calls do not need API keys. Do not commit `.env`.

## Configuration and invocation logs

Edit `config/app.json` to configure the app name, provider, planner and critic models, generation settings, log path, maximum tool steps, and Groq token-rate-limit handling. The default planner model is `openai/gpt-oss-120b`; change either model to an ID available to your Groq account. The tool loop is capped at three successful calls by default, and the planner prompt asks it to call each tool at most once per trip. At the limit, the app disables tools and asks the model to finish with the information already gathered. For TPM rate limits, the UI shows a live countdown and retries after `rateLimitRetryDelaySeconds` (30 seconds by default), up to `maxRateLimitRetries` (one retry by default).

Every Groq request and response, including intermediate tool-call turns, is appended to `logs/invocations.jsonl`. Records include the full messages/tools sent, complete provider response, timestamp, model, status, duration, and token usage. Each external API request is also appended to `logs/api_calls.jsonl`, including tool lookups and geocoding requests; API keys are redacted. These logs contain trip details and generated itineraries, so keep them local and handle them accordingly.

Available planner tools are `search_attractions` (OpenTripMap), `get_weather` (Open-Meteo), and `get_travel_time` (OSRM with OpenStreetMap geocoding for place names). Attraction coordinates can be passed to the routing tool to skip geocoding calls. The critic receives successful tool results as context but has no tool access.

Itinerary prices and local suggestions are estimates. Confirm timings and prices before you travel.
