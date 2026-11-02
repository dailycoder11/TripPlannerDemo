"""One-shot, advisory-only review of an evaluated trip itinerary."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Callable

from groq import Groq


CRITIC_SYSTEM_PROMPT = """You are a Trip Plan Critic.

Your job is to review an already generated and validated trip itinerary and provide concise improvement suggestions.

You are advisory only.

Rules:
1. Do not rewrite or modify the itinerary.
2. Do not create a new itinerary.
3. Do not call or request any tools.
4. Do not change dates, costs, timings or locations.
5. Base your feedback only on the information provided.
6. Focus on practical quality rather than hard validation rules, because deterministic checks have already been completed.
7. Give at most 3 useful suggestions.
8. If the plan is already good, say so instead of inventing problems.

Consider practicality of the schedule, excessive travel, pacing, lunch timing, activity variety,
use of provided weather information, and suitability for the group.

Return only JSON in this structure:
{"overall_feedback":"short assessment","suggestions":["suggestion 1"]}
"""


def review_plan(
    *,
    api_key: str,
    model: str,
    requirements: dict[str, Any],
    plan: dict[str, Any],
    tool_facts: list[dict[str, Any]],
    invocation_id: str,
    call_index: int,
    max_tokens: int,
    temperature: float,
    log_invocation: Callable[[dict[str, Any]], None],
) -> dict[str, Any] | None:
    """Review a validated plan once; return None for call or response failures."""
    user_payload = {
        "original_trip_requirements": requirements,
        "final_validated_plan": plan,
        "available_tool_facts": tool_facts,
    }
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    started = time.monotonic()
    response_data: dict[str, Any] = {}
    call_status = "error"
    feedback: dict[str, Any] | None = None
    error_text: str | None = None
    try:
        client = Groq(api_key=api_key, timeout=45.0, max_retries=0)
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            response_format={"type": "json_object"},
        )
        response_data = completion.model_dump(mode="json")
        content = completion.choices[0].message.content or ""
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Critic response must be a JSON object.")
        overall = parsed.get("overall_feedback")
        suggestions = parsed.get("suggestions")
        if not isinstance(overall, str) or not overall.strip():
            raise ValueError("Critic response is missing overall_feedback.")
        if not isinstance(suggestions, list) or any(not isinstance(item, str) for item in suggestions):
            raise ValueError("Critic suggestions must be a list of strings.")
        feedback = {
            "status": "available",
            "overall_feedback": overall.strip(),
            "suggestions": [item.strip() for item in suggestions if item.strip()][:3],
        }
        call_status = "response_received"
    except Exception as exc:  # Provider and response-format errors should not hide the plan.
        error_text = f"{type(exc).__name__}: {exc}"
        if response_data:
            call_status = "invalid_response"
        diagnostic = error_text
        if api_key:
            diagnostic = diagnostic.replace(api_key, "[redacted]")
        if len(diagnostic) > 300:
            diagnostic = diagnostic[:300] + "...<truncated>"
        print(f"[CRITIC] unavailable ({call_status}): {diagnostic}")
    finally:
        log_invocation({
            "timestamp": datetime.now().astimezone().isoformat(),
            "invocationId": invocation_id,
            "callIndex": call_index,
            "provider": "groq",
            "model": model,
            "status": call_status,
            "durationMs": round((time.monotonic() - started) * 1000),
            "prompt": {
                "messages": messages,
                "tools": None,
                "toolChoice": "none",
                "temperature": temperature,
                "maxTokens": max_tokens,
                "responseFormat": {"type": "json_object"},
            },
            "response": response_data if response_data else ({"error": error_text} if error_text else {}),
        })
    return feedback
