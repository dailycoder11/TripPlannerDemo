# Daytrip



## Phase1: basic-daytrip-planner

A Streamlit day-trip planner that collects trip details, generates an itinerary with a Groq model, and logs model invocations.
A small Streamlit day trip planner. Enter a destination, date, group size, per-person budget, and time window to get an AI-generated itinerary through Groq.

## Run locally

Requires Python 3.10 or newer.

1. Create and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and set `GROQ_API_KEY` to a key from [Groq Console](https://console.groq.com/keys).
4. Start the app: `streamlit run app.py`.

The Groq API key stays server-side and is never written to invocation logs. Do not commit `.env`.

## Configuration and invocation logs

Edit `config/app.json` to configure the app name, provider, model, generation settings, log path, and logging detail. The default model is `openai/gpt-oss-70b`; change it to any model ID available to your Groq account.

Every attempted Groq invocation is appended to the configured JSON Lines log (`logs/invocations.jsonl` by default). Records include a timestamp, request ID, provider, model, status, duration, token usage when supplied, and error details on failure. Trip inputs and model responses are omitted by default. Set `logRequestAndResponse` to `true` in `config/app.json` to include them.

Itinerary prices and local suggestions are estimates. Confirm timings and prices before you travel.
