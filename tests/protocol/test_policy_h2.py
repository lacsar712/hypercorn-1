from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from h2.connection import H2Connection
from h2.events import PushedStreamReceived, StreamReset

from hypercorn.asyncio.worker_context import WorkerContext
from hypercorn.config import Config
from hypercorn.events import RawData
from hypercorn.policy import (
    H2Surface,
    PolicyFrame,
    PolicyTimingError,
    bind_phase,
    current_phase,
    policy_frame,
    push,
)
from hypercorn.protocol.events import Request as StreamRequest
from hypercorn.protocol.h2 import REFUSED_STREAM_ERROR_CODE, H2Protocol
from hypercorn.typing import ConnectionState

HEADERS = [
    (b":method", b"GET"),
    (b":path", b"/"),
    (b":authority", b"hypercorn"),
    (b":scheme", b"https"),
]


def _protocol() -> H2Protocol:
    task_group = AsyncMock()
    # send_task is spawned as a plain task, avoid an un-awaited mock warning.
    task_group.spawn = Mock()
    return H2Protocol(
        Mock(),
        Config(),
        WorkerContext(None),
        task_group,
        ConnectionState({}),
        False,
        None,
        None,
        AsyncMock(),
    )


async def _open_request(
    protocol: H2Protocol, client: H2Connection, stream_id: int, end_stream: bool = False
) -> None:
    client.send_headers(stream_id, HEADERS, end_stream=end_stream)
    await protocol.handle(RawData(data=client.data_to_send()))


def _raw_data(protocol: H2Protocol, skip_preamble: bool = True) -> bytes:
    calls = list(protocol.send.call_args_list)  # type: ignore[attr-defined]
    chunks = [call.args[0].data for call in calls if isinstance(call.args[0], RawData)]
    if skip_preamble:
        # The first chunk is the server SETTINGS preamble from initiate().
        chunks = chunks[1:]
    return b"".join(chunks)


@pytest.mark.asyncio
async def test_deny_push_does_not_emit_push_promise() -> None:
    protocol = _protocol()
    await protocol.initiate()
    protocol._create_server_push = AsyncMock()  # type: ignore[method-assign]
    with bind_phase("connection") as bound, policy_frame(h2_surface=H2Surface.DENY_PUSH):
        await protocol.stream_send(
            StreamRequest(
                stream_id=1,
                headers=[(b":authority", b"hypercorn")],
                http_version="2",
                method="GET",
                raw_path=b"/pushed",
                state=ConnectionState({}),
            )
        )
    protocol._create_server_push.assert_not_awaited()
    assert _raw_data(protocol) == b""
    assert [obs.action for obs in bound.phase.observations] == ["server_push_denied"]


@pytest.mark.asyncio
async def test_honor_multiplex_emits_push() -> None:
    protocol = _protocol()
    await protocol.initiate()
    client = H2Connection()
    client.initiate_connection()
    await _open_request(protocol, client, 1)
    await protocol.stream_send(
        StreamRequest(
            stream_id=1,
            headers=[
                (b":authority", b"hypercorn"),
                (b":scheme", b"https"),
            ],
            http_version="2",
            method="GET",
            raw_path=b"/pushed",
            state=ConnectionState({}),
        )
    )
    assert any(stream_id % 2 == 0 for stream_id in protocol.streams)
    events = client.receive_data(_raw_data(protocol, skip_preamble=False))
    assert any(isinstance(event, PushedStreamReceived) for event in events)


@pytest.mark.asyncio
async def test_deny_push_keeps_concurrent_normal_streams() -> None:
    protocol = _protocol()
    await protocol.initiate()
    client = H2Connection()
    client.initiate_connection()
    client.send_headers(1, HEADERS, end_stream=True)
    client.send_headers(3, HEADERS, end_stream=True)
    with policy_frame(h2_surface=H2Surface.DENY_PUSH):
        await protocol.handle(RawData(data=client.data_to_send()))
    assert set(protocol.streams) == {1, 3}
    # Ordinary streams are not reset or refused.
    events = client.receive_data(_raw_data(protocol))
    assert not any(isinstance(event, StreamReset) for event in events)


@pytest.mark.asyncio
async def test_single_stream_resets_extra_concurrent_stream() -> None:
    protocol = _protocol()
    await protocol.initiate()
    client = H2Connection()
    client.initiate_connection()
    client.send_headers(1, HEADERS, end_stream=False)
    client.send_headers(3, HEADERS, end_stream=True)
    with bind_phase("connection") as bound, policy_frame(h2_surface=H2Surface.SINGLE_STREAM):
        await protocol.handle(RawData(data=client.data_to_send()))
    assert set(protocol.streams) == {1}
    events = client.receive_data(_raw_data(protocol, skip_preamble=False))
    resets = [event for event in events if isinstance(event, StreamReset)]
    assert len(resets) == 1
    assert resets[0].stream_id == 3
    assert resets[0].error_code == REFUSED_STREAM_ERROR_CODE
    assert [obs.action for obs in bound.phase.observations] == ["stream_refused"]


@pytest.mark.asyncio
async def test_single_stream_allows_sequential_streams() -> None:
    protocol = _protocol()
    await protocol.initiate()
    client = H2Connection()
    client.initiate_connection()
    with bind_phase("connection") as bound, policy_frame(h2_surface=H2Surface.SINGLE_STREAM):
        client.send_headers(1, HEADERS, end_stream=True)
        await protocol.handle(RawData(data=client.data_to_send()))
        await protocol._close_stream(1)
        client.send_headers(3, HEADERS, end_stream=True)
        await protocol.handle(RawData(data=client.data_to_send()))
    assert set(protocol.streams) == {3}
    events = client.receive_data(_raw_data(protocol))
    assert not any(isinstance(event, StreamReset) for event in events)
    # Multiplexing never opened, the frame remains valid.
    assert bound.phase.multiplex_opened is False


@pytest.mark.asyncio
async def test_late_single_stream_push_rejected_after_multiplex() -> None:
    protocol = _protocol()
    await protocol.initiate()
    client = H2Connection()
    client.initiate_connection()
    with bind_phase("connection") as bound:
        client.send_headers(1, HEADERS, end_stream=False)
        client.send_headers(3, HEADERS, end_stream=False)
        await protocol.handle(RawData(data=client.data_to_send()))
        assert bound.phase.multiplex_opened is True
        with pytest.raises(PolicyTimingError):
            push(PolicyFrame(h2_surface=H2Surface.SINGLE_STREAM))


@pytest.mark.asyncio
async def test_concurrent_connections_h2_surface_isolation() -> None:
    deny_protocol = _protocol()
    await deny_protocol.initiate()
    honor_protocol = _protocol()
    await honor_protocol.initiate()
    deny_client = H2Connection()
    deny_client.initiate_connection()
    honor_client = H2Connection()
    honor_client.initiate_connection()
    await _open_request(deny_protocol, deny_client, 1)
    await _open_request(honor_protocol, honor_client, 1)

    push_request = StreamRequest(
        stream_id=1,
        headers=[(b":authority", b"hypercorn"), (b":scheme", b"https")],
        http_version="2",
        method="GET",
        raw_path=b"/pushed",
        state=ConnectionState({}),
    )

    async def run_denied() -> None:
        with bind_phase("connection"), policy_frame(h2_surface=H2Surface.DENY_PUSH):
            await deny_protocol.stream_send(push_request)

    async def run_honor() -> None:
        with bind_phase("connection"):
            await honor_protocol.stream_send(push_request)

    await asyncio.gather(run_denied(), run_honor())
    # Only the honouring connection opened a (server pushed) stream.
    assert not any(stream_id % 2 == 0 for stream_id in deny_protocol.streams)
    assert any(stream_id % 2 == 0 for stream_id in honor_protocol.streams)
    assert current_phase("connection") is None
