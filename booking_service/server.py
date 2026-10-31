"""Run the independent ticket-booking A2A agent."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct, Value

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)

from booking_service.logic import process_booking
from booking_service.storage import BookingStore

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
STORAGE_PATH = ROOT / CONFIG.get("storageFile", "bookings.json")
STORE = BookingStore(STORAGE_PATH)
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("ticket_booking")
BOOKING_INPUT_FIELDS = (
    "request_id",
    "event_name",
    "event_date",
    "attendee_count",
    "amount_per_person_inr",
)


class TicketBookingExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            task = Task(
                id=context.task_id or str(uuid.uuid4()),
                context_id=context.context_id or str(uuid.uuid4()),
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(state=TaskState.TASK_STATE_WORKING)
        input_params = {field: None for field in BOOKING_INPUT_FIELDS}
        try:
            request = None
            message = context.message
            if message:
                for part in message.parts:
                    if part.WhichOneof("content") == "data":
                        request = MessageToDict(part.data.struct_value, preserving_proto_field_name=True)
                        break
            if isinstance(request, dict):
                input_params.update({field: request.get(field) for field in BOOKING_INPUT_FIELDS})
            result = await asyncio.to_thread(process_booking, request, CONFIG, STORE)
            await updater.add_artifact(
                parts=[Part(data=Value(struct_value=_as_struct(result)))],
                name="booking_result",
                metadata={"contentType": "application/json"},
            )
            confirmation_id = result.get("confirmation_id", "")
            LOGGER.info(json.dumps({
                "task_id": task.id,
                "status": result.get("status", "failed"),
                "confirmation_id": confirmation_id,
                "input_params": input_params,
            }))
            await updater.update_status(state=TaskState.TASK_STATE_COMPLETED)
        except Exception as exc:
            LOGGER.error(json.dumps({
                "task_id": task.id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "input_params": input_params,
            }))
            failed = {"status": "failed", "error_code": "booking_service_error", "message": "The booking service could not process this request."}
            await updater.add_artifact(parts=[Part(data=Value(struct_value=_as_struct(failed)))], name="booking_result")
            await updater.update_status(state=TaskState.TASK_STATE_COMPLETED)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task:
            updater = TaskUpdater(event_queue, context.current_task.id, context.current_task.context_id)
            await updater.update_status(state=TaskState.TASK_STATE_CANCELED)


def _as_struct(data: dict[str, Any]) -> Struct:
    result = Struct()
    result.update(data)
    return result


def create_app() -> FastAPI:
    base_url = os.getenv("BOOKING_PUBLIC_URL", f"http://{CONFIG['host']}:{CONFIG['port']}")
    card = AgentCard(
        name="Dummy Ticket Booking Agent",
        description=(
            "Daytrip's optional ticket-booking agent, used after a day-trip itinerary is planned. "
            "It checks event requests against the booking service's demo rules, returns a confirmation "
            "for a compliant request, or proposes compliant changes for the user to approve. "
            "It creates local dummy records only; it does not contact ticket providers or process payments."
        ),
        supported_interfaces=[AgentInterface(url=f"{base_url}/", protocol_binding="JSONRPC", protocol_version="1.0")],
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[AgentSkill(
            id="dummy_ticket_booking",
            name="Dummy ticket booking",
            description=(
                "Use after Daytrip has finished planning to validate a ticket or event request. "
                "Input is JSON data with request_id, event_name, event_date, attendee_count, and "
                "amount_per_person_inr. Output is a confirmation, a proposal awaiting user approval, "
                "or a structured failure. Accepting a proposal requires a new A2A task."
            ),
            tags=["ticket", "booking", "dummy"],
            examples=[
                '{"request_id":"demo-123","event_name":"City Museum","event_date":"2026-11-15","attendee_count":2,"amount_per_person_inr":750}'
            ],
            input_modes=["application/json"],
            output_modes=["application/json"],
        )],
    )
    handler = DefaultRequestHandler(
        agent_executor=TicketBookingExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = FastAPI(title="Dummy Ticket Booking Agent")
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"),
    )
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG["host"], port=int(CONFIG["port"]))
