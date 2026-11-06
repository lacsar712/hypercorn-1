from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

import pytest
import trio

from hypercorn.app_wrappers import ASGIWrapper
from hypercorn.config import Config
from hypercorn.policy import LifespanMode, bind_phase, policy_frame
from hypercorn.trio.lifespan import Lifespan
from hypercorn.typing import ASGIReceiveCallable, ASGISendCallable, Scope
from hypercorn.utils import LifespanFailureError, LifespanTimeoutError
from ..helpers import SlowLifespanFramework


@pytest.mark.trio
async def test_startup_timeout_error(nursery: trio._core._run.Nursery) -> None:
    config = Config()
    config.startup_timeout = 0.01
    lifespan = Lifespan(ASGIWrapper(SlowLifespanFramework(0.02, trio.sleep)), config, {})
    nursery.start_soon(lifespan.handle_lifespan)
    with pytest.raises(LifespanTimeoutError) as exc_info:
        await lifespan.wait_for_startup()
    assert str(exc_info.value).startswith("Timeout whilst awaiting startup")


async def _lifespan_failure(
    scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable
) -> None:
    async with trio.open_nursery():
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.failed", "message": "Failure"})
            break


@pytest.mark.trio
async def test_startup_failure() -> None:
    lifespan = Lifespan(ASGIWrapper(_lifespan_failure), Config(), {})
    try:
        async with trio.open_nursery() as lifespan_nursery:
            await lifespan_nursery.start(lifespan.handle_lifespan)
            await lifespan.wait_for_startup()
    except ExceptionGroup as error:
        assert error.subgroup(LifespanFailureError) is not None


async def _startup_then_shutdown_error_app(
    scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable
) -> None:
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            raise RuntimeError("shutdown boom")


@pytest.mark.trio
async def test_policy_suppress_startup_continues() -> None:
    with bind_phase("worker") as bound, policy_frame(lifespan_mode=LifespanMode.SUPPRESS_STARTUP):
        lifespan = Lifespan(ASGIWrapper(_lifespan_failure), Config(), {})
        async with trio.open_nursery() as nursery:
            nursery.start_soon(lifespan.handle_lifespan)
            await lifespan.wait_for_startup()
            nursery.cancel_scope.cancel()
        assert [obs.action for obs in bound.phase.observations] == ["startup_failure_suppressed"]


@pytest.mark.trio
async def test_policy_strict_shutdown_blocks_clean_exit() -> None:
    with bind_phase("worker") as bound, policy_frame(lifespan_mode=LifespanMode.STRICT_SHUTDOWN):
        lifespan = Lifespan(ASGIWrapper(_startup_then_shutdown_error_app), Config(), {})
        with pytest.raises(BaseExceptionGroup) as exc_info:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(lifespan.handle_lifespan)
                await lifespan.wait_for_startup()
                await lifespan.wait_for_shutdown()
        assert exc_info.value.subgroup(RuntimeError) is not None
        assert bound.phase.shutdown_failed is True
        assert bound.phase.clean_exit is False
        assert [obs.action for obs in bound.phase.observations] == ["shutdown_failure_strict"]


@pytest.mark.trio
async def test_policy_honor_shutdown_error_is_clean_baseline() -> None:
    with bind_phase("worker") as bound:
        lifespan = Lifespan(ASGIWrapper(_startup_then_shutdown_error_app), Config(), {})
        async with trio.open_nursery() as nursery:
            nursery.start_soon(lifespan.handle_lifespan)
            await lifespan.wait_for_startup()
            # Existing behaviour: the error is logged and shutdown completes.
            await lifespan.wait_for_shutdown()
        assert bound.phase.clean_exit is True
