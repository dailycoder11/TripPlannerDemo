# Daytrip

Daytrip is a Streamlit day-trip planner. Its itinerary flow uses Groq and travel-data tools. **Phase 9** adds a separate local Policy Service where users can search travel-company and public policy documents and inspect cited excerpts. **Phase 10** adds a local service that collects and compares tourism news for five configured Indian cities. Policy search and news collection run as services, separate from the itinerary agent.

## Phase 9: Policy Search (RAG)

The Policy Service reads PDF documents from `policy_service/documents/`, extracts page text, creates local embeddings, and searches those passages by semantic similarity. Results show the source title, issuer, page, excerpt, source link, and available date/context information. This helps users find source material; it is not legal advice, a compliance assessment, or a statement of Daytrip policy. Verify time-sensitive information with the source publisher.

The active corpus has nine sources: IRCTC refund rules, protected-monument visitor rules, the DPDP Act, Goibibo terms and privacy policy, redBus terms, privacy policy and booking FAQ, and the internally authored **Daytrip Planner: Product Policy and Responsible Use Guide**. The five company webpages were captured as searchable PDFs on 1 November 2026 and include their original source URLs. General consumer e-commerce rules, dark-pattern guidance, and MakeMyTrip's CCPA self-audit have been removed from the repository because they are outside this planner's focused policy scope. The travel-company snapshots can become stale, and supplier-specific terms may control a particular ticket or booking.

The internal guide is based on official government and standards sources. It covers truthful trip suggestions, price/route/weather uncertainty, consumer choices and booking hand-offs, privacy, safety, accessibility, AI/RAG controls, operations, and a release checklist. It distinguishes practical product recommendations from legal duties and includes a linked reference register. Its PDF is at [policy_service/documents/daytrip_planner_policy_guide_2026.pdf](/Users/abhi/RND/BMS/DailyPlannerDemo/policy_service/documents/daytrip_planner_policy_guide_2026.pdf). Source metadata for active and retained files is in `policy_service/sources.json`.

### Start the app and service

Use Python 3.10 or newer. From the repository root, create the virtual environment and install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first service startup downloads the public embedding model `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face and caches it on the machine. No Hugging Face token is needed. On a machine without Internet after that first download, start the service in offline mode so the Hugging Face libraries use the cache without checking the Hub:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m policy_service.server
```

With the model cached, question embedding and search run locally; ask requests do not need Internet access.

Start the Policy Service in Terminal 1:

```sh
.venv/bin/python -m policy_service.server
```

Wait for the startup log to report `policy_index_ready`. The service extracts the PDFs, embeds the passages, and creates an in-memory FAISS index each time it starts. Check readiness at [http://127.0.0.1:8002/health](http://127.0.0.1:8002/health).

Start Daytrip in Terminal 2:

```sh
.venv/bin/streamlit run app.py
```

Set the existing planner credentials in `.env` if you also want itinerary planning. Policy lookup itself does not require a Groq key. Copy `.env.example` to `.env` as a starting point; keep `.env` private and do not commit it.

### Ask a policy question

Open the **Ask about policy** panel in the sidebar, enter a question such as “What does the IRCTC document say about ticket cancellation?”, and submit it. This separate form does not submit or clear the trip-planning form. The sidebar shows up to the two highest-ranked excerpts with readable source details, an official source link, and a similarity score labeled as retrieval confidence. That score is not a probability that a passage is correct. A no-match message means none of the indexed passages cleared the configured similarity threshold. The service returns source passages rather than composing a natural-language answer.

You can call the service directly:

```sh
curl -X POST http://127.0.0.1:8002/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"When can an IRCTC ticket be cancelled?"}'
```

The request contains one non-empty `question` string of up to 1,200 characters. Unknown fields are rejected. A successful response has this shape:

```json
{
  "question": "When can an IRCTC ticket be cancelled?",
  "match_status": "found",
  "matches": [
    {
      "document_id": "irctc-refund-cancellation-rules",
      "document_title": "Refund and Cancellation Rules for IRCTC Trains",
      "issuer": "Indian Railway Catering and Tourism Corporation (IRCTC)",
      "category": "travel",
      "family": "Indian Railways ticket refunds",
      "source_url": "https://contents.irctc.co.in/en/RefundCancellationRules.pdf",
      "published_date": null,
      "effective_date": null,
      "validity_note": "Check the official IRCTC source for current updates before relying on time-sensitive fares or cancellation rules.",
      "retrieved_at": "2026-11-01",
      "page_numbers": [1],
      "similarity_score": 0.61,
      "excerpt": "A matching passage extracted from the cited PDF page."
    }
  ],
  "message": "Retrieved source passages from local policy-source documents. No generated interpretation was used."
}
```

`match_status` is `no_match` with an empty `matches` array when nothing meets the threshold. The service returns at most two matches. Each match has a `similarity_score` (cosine similarity over normalized embeddings). The service logs each returned match's document ID, page, and score, without logging the question. `GET /health` reports readiness, document and passage counts, model name, and non-fatal document warnings. If the index is unavailable, the endpoint returns HTTP 503 and the UI reports that the service is not ready.

### Configuration and boundaries

- `policy_service/config.json` controls the bind host and port, embedding model, CPU device, chunk size and overlap, similarity threshold, and result limit (capped at two).
- The service URL defaults to `http://127.0.0.1:8002`. Change `policyServiceUrl` in `config/app.json` or set `DAYTRIP_POLICY_URL` in the environment.
- The service binds to loopback and has no authentication. Keep it local; do not expose it on a network without adding appropriate access controls.
- There is no LLM call or web search in the question path. The model is used only to embed the question locally and find similar indexed passages.
- To add or replace sources, put verified PDFs in `policy_service/documents/`, update `policy_service/sources.json` with accurate source metadata and validity notes, then restart the Policy Service. It rebuilds the index at startup. Files listed under `retained_not_indexed` remain on disk but are not included in retrieval.
- Individual unreadable or missing PDFs are skipped and reported as warnings if other documents can be indexed. If no PDF text can be extracted, startup fails and health is unavailable.

## Phase 10: city tourism news

The local News Service searches Tavily for tourism news about the five cities in `news_service/config.json` (Shimla, Manali, Jaipur, Agra, and Udaipur). It stores article titles, short excerpts, publication dates, and original source URLs in the file-backed SQLite database `news_service/data/news.sqlite3`. The database can be browsed in VS Code with a SQLite viewer extension.

Each startup searches the previous twelve completed calendar months in four consecutive three-month windows, plus the latest window from one year earlier for comparison. On 1 November 2026, those windows are November 2025–January 2026, February–April 2026, May–July 2026, August–October 2026, and the comparison August–October 2025. The service makes 25 basic searches per successful refresh (five cities times five windows), with up to ten results per search. Tavily's free plan provides 1,000 credits per month; basic searches cost one credit and the free key does not require a credit card. Search access requires a Tavily API key. Keep it server-side in `.env` as `TAVILY_API_KEY`; do not commit it.

Start the News Service in its own terminal:

```sh
.venv/bin/python -m news_service.server
```

The service binds to `127.0.0.1:8003` by default. Readiness and refresh state are available at [http://127.0.0.1:8003/health](http://127.0.0.1:8003/health). The app calls `GET /api/v1/news?city=Shimla`; its service URL defaults to `http://127.0.0.1:8003` and can be overridden with `DAYTRIP_NEWS_URL`.

The collapsible **Tourism news** panel uses the planner's destination field. For configured cities it shows the latest completed three-month window and the same months one year earlier; the selected trip date does not change the period. Unsupported destinations leave the panel empty. If Tavily fails during startup, the service keeps and serves the last complete database snapshot. A first refresh failure leaves the service unavailable until a refresh succeeds.

## Run Daytrip

The planner requires Python 3.10 or newer and the dependencies above. Copy `.env.example` to `.env` and set `GROQ_API_KEY` and `OPENTRIPMAP_API_KEY` for itinerary planning. Add `TAVILY_API_KEY` to enable Phase 10 tourism news. Then run:

```sh
.venv/bin/streamlit run app.py
```

Use the form to enter a destination, date, group size, budget, and time window. The planner can use attraction, weather, and route tools to prepare an itinerary. Prices and local suggestions are estimates; confirm timings and prices before travel.

## Phase 7: A2A ticket-booking demo

Daytrip and the dummy ticket-booking service run as separate processes. Start the booking service in one terminal and Daytrip in another:

```sh
.venv/bin/python -m booking_service.server
```

```sh
.venv/bin/streamlit run app.py
```

The A2A endpoint defaults to `http://127.0.0.1:8001`; configure it with `bookingServiceUrl` in `config/app.json` or `DAYTRIP_BOOKING_URL`. Service host, port, model, JSON storage path, and booking policy values are in `booking_service/config.json`.

Booking requests contain `request_id`, `event_name`, `event_date`, `attendee_count`, and `amount_per_person_inr`. Results can be confirmed, proposed, or failed. A proposal requires user acceptance before a new booking request is sent. The service writes task outcomes to its process output and local JSON storage; credentials are excluded.

## Phase 8: trip-request safety screening

Before memory lookup or itinerary planning, Daytrip screens submitted trip fields with the configured Groq screening model. Suspicious or unrelated instructions, malformed screening output, validation failures, and screening-service errors stop the request. The app also validates tool arguments and external results before using them. These controls reduce risk but cannot guarantee that every prompt-injection attempt will be detected.

## Configuration and logs

Edit `config/app.json` to configure the planner, screening and critic models, generation settings, service URLs, and logging behavior. Set model IDs to models available to your Groq account. Groq and OpenTripMap credentials stay in environment configuration; do not commit `.env`.

Invocation and API logs are stored under `logs/` and may contain trip details or generated itineraries. Keep them local and handle them accordingly. Open-Meteo, OpenStreetMap Nominatim, and OSRM calls do not need API keys.
