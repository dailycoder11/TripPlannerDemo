"""A2A client helpers used by the Streamlit Daytrip UI."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
from google.protobuf.struct_pb2 import Struct, Value

from a2a.client import ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, SendMessageRequest


class BookingServiceError(RuntimeError):
    pass


async def _send(url: str, request_data: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    http_client = httpx.AsyncClient(timeout=12.0)
    try:
        client = await ClientFactory(ClientConfig(httpx_client=http_client)).create_from_url(
            url,
            resolver_http_kwargs={"timeout": 12.0},
        )
    except Exception:
        await http_client.aclose()
        raise
    struct = Struct()
    struct.update(request_data)
    message = Message(
        message_id=str(uuid.uuid4()),
        role=Role.ROLE_USER,
        parts=[Part(data=Value(struct_value=struct))],
    )
    task_id = None
    result = None
    try:
        stream = client.send_message(SendMessageRequest(message=message))
        async for response in stream:
            if response.WhichOneof("payload") == "task":
                task = response.task
                task_id = task.id or task_id
                for artifact in task.artifacts:
                    for part in artifact.parts:
                        if part.WhichOneof("content") == "data":
                            from google.protobuf.json_format import MessageToDict
                            result = MessageToDict(part.data.struct_value, preserving_proto_field_name=True)
            elif response.WhichOneof("payload") == "status_update" and response.status_update.task_id:
                task_id = response.status_update.task_id
        if not isinstance(result, dict) or result.get("status") not in {"confirmed", "proposal", "failed"}:
            raise BookingServiceError("The booking service returned an incomplete or malformed A2A result.")
        return result, task_id
    finally:
        await client.close()


def send_booking_request(url: str, request_data: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Send one A2A task synchronously from Streamlit's script thread."""
    try:
        return asyncio.run(_send(url, request_data))
    except BookingServiceError:
        raise
    except Exception as exc:
        raise BookingServiceError("The booking service is unavailable or returned an invalid A2A response.") from exc
