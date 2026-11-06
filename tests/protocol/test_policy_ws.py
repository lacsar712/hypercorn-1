from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import AsyncMock, call

import pytest

from hypercorn.app_wrappers import ASGIWrapper
from hypercorn.asyncio.task_group import TaskGroup
from hypercorn.asyncio.worker_context import WorkerContext
from hypercorn.config import Config
from hypercorn.policy import (
    PolicyFrame,
    PolicyTimingError,
    WSMode,
    bind_phase,
    observations,
    policy_frame,
    push,
)
from hypercorn.protocol.events import (
    Data,
    EndBody,
    Request,
    Response,
    StreamClosed,
)
from hypercorn.protocol.ws_stream import ASGIWebsocketState, WSStream
from hypercorn.typing import ConnectionState, WebsocketAcceptEvent
from hypercorn.typing import ASGIReceiveCallable, ASGISendCallable, Scope

WS_HEADERS = [
    (b"connection", b"upgrade, keep-alive"),
    (b"sec-websocket-version", b"13"),
    (b"upgrade", b"websocket"),
    (b"sec-websocket-key", b"UnQ3lpJAH6j2PslA993iKQ=="),
]
WS2_HEADERS = [(b"sec-websocket-version", b"13")]


def _mocked_stream() -> WSStream:
    stream = WSStream(
        AsyncMock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        False,
        None,
        None,
        AsyncMock(),
        1,
    )
    stream.task_group.spawn_app.return_value = AsyncMock()  # type: ignore[attr-defined]
    stream.app_put = AsyncMock()
    return stream


def _request(http_version: str, headers: list[tuple[bytes, bytes]]) -> Request:
    return Request(
        stream_id=1,
        http_version=http_version,
        headers=headers,
        raw_path=b"/",
        method="GET",
        state=ConnectionState({}),
    )


@pytest.mark.asyncio
async def test_deny_upgrade_rejects_handshake_with_http_error() -> None:
    stream = _mocked_stream()
    with bind_phase("connection") as bound, policy_frame(ws_mode=WSMode.DENY_UPGRADE):
        await stream.handle(_request("1.1", WS_HEADERS))
    assert stream.closed is True
    stream.task_group.spawn_app.assert_not_called()  # type: ignore[attr-defined]
    assert stream.send.call_args_list == [  # type: ignore[attr-defined]
        call(
            Response(
                stream_id=1,
                headers=[(b"content-length", b"0"), (b"connection", b"close")],
                status_code=403,
            )
        ),
        call(EndBody(stream_id=1)),
    ]
    assert [obs.action for obs in bound.phase.observations] == ["upgrade_denied"]


@pytest.mark.asyncio
async def test_deny_upgrade_at_accept_does_not_open_session() -> None:
    stream = _mocked_stream()
    with policy_frame(ws_mode=WSMode.HONOR_WS):
        await stream.handle(_request("1.1", WS_HEADERS))
    with bind_phase("connection") as bound, policy_frame(ws_mode=WSMode.DENY_UPGRADE):
        await stream.app_send(cast(WebsocketAcceptEvent, {"type": "websocket.accept"}))
    assert stream.state is ASGIWebsocketState.HTTPCLOSED
    assert stream.closed is True
    assert stream.send.call_args_list[0].args[0].status_code == 403  # type: ignore[attr-defined]
    assert [obs.action for obs in bound.phase.observations] == ["upgrade_denied"]


@pytest.mark.asyncio
async def test_late_deny_upgrade_after_accept_rejected() -> None:
    stream = _mocked_stream()
    with bind_phase("connection"):
        await stream.handle(_request("1.1", WS_HEADERS))
        await stream.app_send(cast(WebsocketAcceptEvent, {"type": "websocket.accept"}))
        with pytest.raises(PolicyTimingError):
            push(PolicyFrame(ws_mode=WSMode.DENY_UPGRADE))


async def _crashing_app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
    await send(cast(WebsocketAcceptEvent, {"type": "websocket.accept"}))
    raise RuntimeError("boom")


async def _drive_until_closed(stream: WSStream) -> None:
    for _ in range(100):
        if stream.closed:
            return
        await asyncio.sleep(0)
    raise AssertionError("stream did not close")


@pytest.mark.asyncio
async def test_close_on_exc_closes_connected_session() -> None:
    event_loop = asyncio.get_running_loop()
    send = AsyncMock()
    async with TaskGroup(event_loop) as task_group:
        stream = WSStream(
            ASGIWrapper(_crashing_app),
            Config(),
            WorkerContext(None),
            task_group,
            False,
            None,
            None,
            send,
            1,
        )
        with bind_phase("connection") as bound, policy_frame(ws_mode=WSMode.CLOSE_ON_EXC):
            await stream.handle(_request("2", WS2_HEADERS))
            await _drive_until_closed(stream)
            assert stream.state is ASGIWebsocketState.CLOSED
            assert [obs.action for obs in bound.phase.observations] == ["closed_on_exception"]
            assert send.call_args_list[0].args[0] == Response(  # accept happened
                stream_id=1, headers=[], status_code=200
            )
            assert Data(stream_id=1, data=b"\x88\x02\x03\xf3") in [
                call.args[0] for call in send.call_args_list
            ]
            assert StreamClosed(stream_id=1) in [call.args[0] for call in send.call_args_list]


@pytest.mark.asyncio
async def test_honor_ws_leaves_session_state_after_app_error() -> None:
    # The observable fork: honouring the stack takes the generic ASGI
    # handler path which sends the close frame but leaves the stream
    # nominally connected and records no policy observation.
    event_loop = asyncio.get_running_loop()
    send = AsyncMock()
    async with TaskGroup(event_loop) as task_group:
        stream = WSStream(
            ASGIWrapper(_crashing_app),
            Config(),
            WorkerContext(None),
            task_group,
            False,
            None,
            None,
            send,
            1,
        )
        with bind_phase("connection"):
            await stream.handle(_request("2", WS2_HEADERS))
            await _drive_until_close_data(send)
            assert stream.state is ASGIWebsocketState.CONNECTED
            assert observations() == ()
            assert Data(stream_id=1, data=b"\x88\x02\x03\xf3") in [
                call.args[0] for call in send.call_args_list
            ]


async def _drive_until_close_data(send: AsyncMock) -> None:
    expected = Data(stream_id=1, data=b"\x88\x02\x03\xf3")
    for _ in range(100):
        if expected in [call.args[0] for call in send.call_args_list]:
            return
        await asyncio.sleep(0)
    raise AssertionError("close frame not sent")
