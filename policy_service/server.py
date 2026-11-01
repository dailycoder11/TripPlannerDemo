"""Run the local, retrieval-only policy service."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from policy_service.index import PolicyIndex

SERVICE_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((SERVICE_DIR / "config.json").read_text(encoding="utf-8"))
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("policy_service")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1200)


class SourceMatch(BaseModel):
    document_id: str
    document_title: str
    issuer: str
    category: str
    family: str
    source_url: str
    published_date: str | None
    effective_date: str | None
    validity_note: str
    retrieved_at: str
    page_numbers: list[int]
    similarity_score: float
    excerpt: str


class AskResponse(BaseModel):
    question: str
    match_status: Literal["found", "no_match"]
    matches: list[SourceMatch]
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.policy_index = None
    app.state.index_error = None
    try:
        app.state.policy_index = await asyncio.to_thread(PolicyIndex.load, SERVICE_DIR, CONFIG)
        LOGGER.info(json.dumps({
            "event": "policy_index_ready",
            "documents": app.state.policy_index.document_count,
            "chunks": len(app.state.policy_index.passages),
            "model": app.state.policy_index.model_name,
            "warnings": app.state.policy_index.warnings,
        }, ensure_ascii=False))
    except Exception as exc:
        app.state.index_error = type(exc).__name__
        LOGGER.exception("Policy index initialization failed (%s)", type(exc).__name__)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Daytrip Policy Search",
        description=(
            "Local semantic search over travel-company and public policy PDF sources. Returns cited source passages; "
            "it does not call a text-generation model or make live web requests for questions."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health(request: Request):
        policy_index = getattr(request.app.state, "policy_index", None)
        if policy_index is None:
            raise HTTPException(status_code=503, detail="Policy index is unavailable.")
        return {
            "status": "ready",
            "documents_indexed": policy_index.document_count,
            "chunks_indexed": len(policy_index.passages),
            "model": policy_index.model_name,
            "warnings": policy_index.warnings,
        }

    @app.post("/api/v1/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request):
        policy_index = getattr(request.app.state, "policy_index", None)
        if policy_index is None:
            raise HTTPException(status_code=503, detail="Policy index is unavailable.")
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be blank.")
        matches = await asyncio.to_thread(policy_index.search, question)
        return AskResponse(
            question=question,
            match_status="found" if matches else "no_match",
            matches=matches,
            message=(
                "Retrieved source passages from local policy-source documents. No generated interpretation was used."
                if matches else
                "No sufficiently similar passages were found in the local policy documents."
            ),
        )

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.get("host", "127.0.0.1"), port=int(CONFIG.get("port", 8002)))
