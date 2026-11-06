from __future__ import annotations

import asyncio

import pytest

from hypercorn.app_wrappers import ASGIWrapper
from hypercorn.asyncio.tcp_server import TCPServer
from hypercorn.asyncio.worker_context import WorkerContext
from hypercorn.config import Config
from hypercorn.policy import KeepPolicy, policy_frame
from hypercorn.typing import ASGIReceiveCallable, ASGISendCallable, Scope
from .helpers import MemoryReader, MemoryWriter

REQUEST = b"GET / HTTP/1.1\r\nHost: hypercorn\r\n\r\n"


async def immediate_framework(
    scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable
) -> None:
    while True:
        event = await receive()
        if event["type"] == "http.disconnect":
            break
        elif event["type"] == "http.request" and not event.get("more_body", False):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-length", b"0")],
                }
            )
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            break


async def _run_with_policy(server: TCPServer, keep: KeepPolicy | None) -> None:
    if keep is None:
        await server.run()
    else:
        with policy_frame(keep_policy=keep):
            await server.run()


async def _wait_closed(writer: MemoryWriter) -> bool:
    for _ in range(100):
        if writer.is_closed:
            return True
        await asyncio.sleep(0.01)
    return writer.is_closed


@pytest.mark.asyncio
async def test_force_close_frame_closes_connection_end_to_end() -> None:
    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    reader = MemoryReader()
    writer = MemoryWriter()
    server = TCPServer(
        ASGIWrapper(immediate_framework),
        event_loop,
        Config(),
        WorkerContext(None),
        {},
        reader,
        writer,
    )
    task = event_loop.create_task(_run_with_policy(server, KeepPolicy.FORCE_CLOSE))
    await reader.send(REQUEST)
    assert await _wait_closed(writer) is True
    reader.close()
    await task


@pytest.mark.asyncio
async def test_empty_stack_keeps_connection_open_end_to_end() -> None:
    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    reader = MemoryReader()
    writer = MemoryWriter()
    server = TCPServer(
        ASGIWrapper(immediate_framework),
        event_loop,
        Config(),
        WorkerContext(None),
        {},
        reader,
        writer,
    )
    task = event_loop.create_task(_run_with_policy(server, None))
    await reader.send(REQUEST)
    for _ in range(20):
        await asyncio.sleep(0.01)
    # No frame pushed - the existing keep-alive behaviour is unchanged.
    assert writer.is_closed is False
    reader.close()
    await task
