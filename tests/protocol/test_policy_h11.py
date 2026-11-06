from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

import hypercorn.protocol.h11
from hypercorn.asyncio.worker_context import EventWrapper
from hypercorn.config import Config
from hypercorn.events import Closed, RawData, Updated
from hypercorn.policy import (
    KeepPolicy,
    PolicyFrame,
    PolicyTimingError,
    bind_phase,
    current_phase,
    policy_frame,
    push,
)
from hypercorn.protocol.events import EndBody
from hypercorn.protocol.events import Response as StreamResponse
from hypercorn.protocol.events import StreamClosed
from hypercorn.protocol.h11 import H11Protocol
from hypercorn.protocol.http_stream import HTTPStream
from hypercorn.typing import ConnectionState

GET_REQUEST = b"GET / HTTP/1.1\r\nHost: hypercorn\r\n\r\n"
POST_PARTIAL = b"POST / HTTP/1.1\r\nHost: hypercorn\r\nContent-Length: 5\r\n\r\nab"
POST_REMAINDER = b"cde"


def _protocol(monkeypatch: MonkeyPatch) -> H11Protocol:
    mock_stream = Mock()
    mock_stream.return_value = AsyncMock(spec=HTTPStream)
    monkeypatch.setattr(hypercorn.protocol.h11, "HTTPStream", mock_stream)
    context = Mock()
    context.event_class.side_effect = EventWrapper
    context.mark_request = AsyncMock()
    context.terminated = EventWrapper()
    return H11Protocol(
        AsyncMock(),
        Config(),
        context,
        AsyncMock(),
        ConnectionState({}),
        False,
        None,
        None,
        AsyncMock(),
    )


def _sent_events(protocol: H11Protocol) -> list:
    return [call.args[0] for call in protocol.send.call_args_list]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_force_close_closes_keepalive_connection(monkeypatch: MonkeyPatch) -> None:
    protocol = _protocol(monkeypatch)
    with bind_phase("connection") as bound, policy_frame(keep_policy=KeepPolicy.FORCE_CLOSE):
        await protocol.handle(RawData(data=GET_REQUEST))
        await protocol.stream_send(StreamResponse(stream_id=1, status_code=200, headers=[]))
        await protocol.stream_send(EndBody(stream_id=1))
        await protocol.stream_send(StreamClosed(stream_id=1))

    events = _sent_events(protocol)
    assert any(isinstance(event, Closed) for event in events)
    assert not any(isinstance(event, Updated) and event.idle for event in events)
    raw = b"".join(event.data for event in events if isinstance(event, RawData))
    assert b"connection: close" in raw.lower()
    actions = [obs.action for obs in bound.phase.observations]
    assert "force_close" in actions


@pytest.mark.asyncio
async def test_honor_keepalive_recycles(monkeypatch: MonkeyPatch) -> None:
    protocol = _protocol(monkeypatch)
    await protocol.handle(RawData(data=GET_REQUEST))
    await protocol.stream_send(StreamResponse(stream_id=1, status_code=200, headers=[]))
    await protocol.stream_send(EndBody(stream_id=1))
    await protocol.stream_send(StreamClosed(stream_id=1))

    events = _sent_events(protocol)
    assert any(isinstance(event, Updated) and event.idle for event in events)
    assert not any(isinstance(event, Closed) for event in events)


@pytest.mark.asyncio
async def test_drain_then_close_waits_for_unread_body(monkeypatch: MonkeyPatch) -> None:
    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    protocol = _protocol(monkeypatch)
    with bind_phase("connection") as bound, policy_frame(keep_policy=KeepPolicy.DRAIN_THEN_CLOSE):
        await protocol.handle(RawData(data=POST_PARTIAL))
        await protocol.stream_send(StreamResponse(stream_id=1, status_code=200, headers=[]))
        await protocol.stream_send(EndBody(stream_id=1))
        finalize = event_loop.create_task(protocol.stream_send(StreamClosed(stream_id=1)))
        await asyncio.sleep(0)
        # Response written, request body not fully received - must not close.
        assert not finalize.done()
        assert not any(isinstance(event, Closed) for event in _sent_events(protocol))

        # Drain the rest of the unread request body, then it closes.
        await protocol.handle(RawData(data=POST_REMAINDER))
        await asyncio.sleep(0)
        assert finalize.done()
        assert any(isinstance(event, Closed) for event in _sent_events(protocol))

    assert [obs.action for obs in bound.phase.observations] == ["drain_then_close"]


@pytest.mark.asyncio
async def test_honor_closes_immediately_with_unread_body(monkeypatch: MonkeyPatch) -> None:
    # The fork from drain_then_close: honour closes as soon as the response
    # ends, it does not wait for an unread body.
    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    protocol = _protocol(monkeypatch)
    await protocol.handle(RawData(data=POST_PARTIAL))
    await protocol.stream_send(StreamResponse(stream_id=1, status_code=200, headers=[]))
    await protocol.stream_send(EndBody(stream_id=1))
    finalize = event_loop.create_task(protocol.stream_send(StreamClosed(stream_id=1)))
    await asyncio.sleep(0)
    assert finalize.done()
    assert any(isinstance(event, Closed) for event in _sent_events(protocol))


@pytest.mark.asyncio
async def test_drain_then_close_when_body_already_complete(monkeypatch: MonkeyPatch) -> None:
    protocol = _protocol(monkeypatch)
    await protocol.handle(
        RawData(data=(b"POST / HTTP/1.1\r\nHost: hypercorn\r\n" b"Content-Length: 5\r\n\r\nabcde"))
    )
    with bind_phase("connection"), policy_frame(keep_policy=KeepPolicy.DRAIN_THEN_CLOSE):
        await protocol.stream_send(StreamResponse(stream_id=1, status_code=200, headers=[]))
        await protocol.stream_send(EndBody(stream_id=1))
        await protocol.stream_send(StreamClosed(stream_id=1))
    # Body already fully drained - close immediately without parking.
    assert any(isinstance(event, Closed) for event in _sent_events(protocol))


@pytest.mark.asyncio
async def test_late_force_close_push_is_rejected(monkeypatch: MonkeyPatch) -> None:
    protocol = _protocol(monkeypatch)
    with bind_phase("connection"):
        await protocol.handle(RawData(data=GET_REQUEST))
        await protocol.stream_send(StreamResponse(stream_id=1, status_code=200, headers=[]))
        await protocol.stream_send(EndBody(stream_id=1))
        await protocol.stream_send(StreamClosed(stream_id=1))
        with pytest.raises(PolicyTimingError):
            push(PolicyFrame(keep_policy=KeepPolicy.FORCE_CLOSE))


@pytest.mark.asyncio
async def test_concurrent_connections_keep_policy_isolation(
    monkeypatch: MonkeyPatch,
) -> None:
    force_protocol = _protocol(monkeypatch)
    honor_protocol = _protocol(monkeypatch)

    async def run(protocol: H11Protocol, keep: KeepPolicy | None) -> list:
        with bind_phase("connection"):
            if keep is not None:
                cm = policy_frame(keep_policy=keep)
                cm.__enter__()
            else:
                cm = None
            try:
                await protocol.handle(RawData(data=GET_REQUEST))
                await protocol.stream_send(StreamResponse(stream_id=1, status_code=200, headers=[]))
                await protocol.stream_send(EndBody(stream_id=1))
                await protocol.stream_send(StreamClosed(stream_id=1))
            finally:
                if cm is not None:
                    cm.__exit__(None, None, None)
        return _sent_events(protocol)

    force_events, honor_events = await asyncio.gather(
        run(force_protocol, KeepPolicy.FORCE_CLOSE),
        run(honor_protocol, None),
    )
    assert any(isinstance(event, Closed) for event in force_events)
    assert not any(isinstance(event, Updated) and event.idle for event in force_events)
    assert any(isinstance(event, Updated) and event.idle for event in honor_events)
    assert not any(isinstance(event, Closed) for event in honor_events)
    # The two connections must not read each other's connection phase.
    assert current_phase("connection") is None
