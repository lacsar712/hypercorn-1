from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from functools import partial
from typing import Any

from .. import policy
from ..config import Config
from ..typing import AppWrapper, ASGIReceiveEvent, ASGISendEvent, LifespanScope, LifespanState
from ..utils import LifespanFailureError, LifespanTimeoutError

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup


class UnexpectedMessageError(Exception):
    pass


class Lifespan:
    def __init__(
        self,
        app: AppWrapper,
        config: Config,
        loop: asyncio.AbstractEventLoop,
        lifespan_state: LifespanState,
    ) -> None:
        self.app = app
        self.config = config
        self.startup = asyncio.Event()
        self.shutdown = asyncio.Event()
        self.app_queue: asyncio.Queue = asyncio.Queue(config.max_app_queue_size)
        self.supported = True
        self.loop = loop
        self.state = lifespan_state

        # This mimics the Trio nursery.start task_status and is
        # required to ensure the support has been checked before
        # waiting on timeouts.
        self._started = asyncio.Event()

    async def handle_lifespan(self) -> None:
        self._started.set()
        scope: LifespanScope = {
            "type": "lifespan",
            "asgi": {"spec_version": "2.0", "version": "3.0"},
            "state": self.state,
        }

        def _call_soon(func: Callable, *args: Any) -> Any:
            future = asyncio.run_coroutine_threadsafe(func(*args), self.loop)
            return future.result()

        try:
            await self.app(
                scope,
                self.asgi_receive,
                self.asgi_send,
                partial(self.loop.run_in_executor, None),
                _call_soon,
            )
        except (LifespanFailureError, asyncio.CancelledError):
            raise
        except (BaseExceptionGroup, Exception) as error:
            if isinstance(error, BaseExceptionGroup):
                reraise_error = error.subgroup((LifespanFailureError, asyncio.CancelledError))
                if reraise_error is not None:
                    raise reraise_error

            self.supported = False
            mode = policy.current_policy(self.config).lifespan_mode
            run_phase = policy.current_phase("worker", create=True)
            assert run_phase is not None
            if not self.startup.is_set():
                run_phase.startup_decided = True
                if mode is policy.LifespanMode.SUPPRESS_STARTUP:
                    run_phase.observe(
                        "lifespan_mode", "startup_failure_suppressed", "framework_exception"
                    )
                    await self.config.log.warning(
                        "ASGI Framework Lifespan startup error suppressed by policy, continuing "
                        "without Lifespan support"
                    )
                else:
                    await self.config.log.warning(
                        "ASGI Framework Lifespan error, continuing without Lifespan support"
                    )
            elif not self.shutdown.is_set():
                if mode is policy.LifespanMode.STRICT_SHUTDOWN:
                    run_phase.shutdown_failed = True
                    run_phase.clean_exit = False
                    run_phase.observe(
                        "lifespan_mode", "shutdown_failure_strict", "framework_exception"
                    )
                    await self.config.log.error(
                        "ASGI Framework Lifespan shutdown error, policy strict_shutdown prevents "
                        "a clean worker exit"
                    )
                    raise
                await self.config.log.exception(
                    "ASGI Framework Lifespan error, shutdown without Lifespan support"
                )
            else:
                await self.config.log.exception("ASGI Framework Lifespan errored after shutdown.")
        finally:
            self.startup.set()
            self.shutdown.set()

    async def wait_for_startup(self) -> None:
        await self._started.wait()
        if not self.supported:
            return

        await self.app_queue.put({"type": "lifespan.startup"})
        try:
            await asyncio.wait_for(self.startup.wait(), timeout=self.config.startup_timeout)
        except asyncio.TimeoutError as error:
            if (
                policy.current_policy(self.config).lifespan_mode
                is policy.LifespanMode.SUPPRESS_STARTUP
            ):
                run_phase = policy.current_phase("worker", create=True)
                assert run_phase is not None
                run_phase.startup_decided = True
                run_phase.observe("lifespan_mode", "startup_failure_suppressed", "timeout")
                await self.config.log.warning(
                    "Lifespan startup timeout suppressed by policy, continuing without a startup"
                )
                return
            raise LifespanTimeoutError("startup") from error

    async def wait_for_shutdown(self) -> None:
        await self._started.wait()
        if not self.supported:
            return

        await self.app_queue.put({"type": "lifespan.shutdown"})
        try:
            await asyncio.wait_for(self.shutdown.wait(), timeout=self.config.shutdown_timeout)
        except asyncio.TimeoutError as error:
            raise LifespanTimeoutError("shutdown") from error

    async def asgi_receive(self) -> ASGIReceiveEvent:
        return await self.app_queue.get()

    async def asgi_send(self, message: ASGISendEvent) -> None:
        if message["type"] == "lifespan.startup.complete":
            self.startup.set()
        elif message["type"] == "lifespan.shutdown.complete":
            self.shutdown.set()
        elif message["type"] == "lifespan.startup.failed":
            self.startup.set()
            run_phase = policy.current_phase("worker", create=True)
            assert run_phase is not None
            run_phase.startup_decided = True
            if (
                policy.current_policy(self.config).lifespan_mode
                is policy.LifespanMode.SUPPRESS_STARTUP
            ):
                run_phase.observe(
                    "lifespan_mode", "startup_failure_suppressed", message.get("message", "")
                )
                await self.config.log.warning(
                    "Lifespan startup failure suppressed by policy: "
                    f"{message.get('message', '')}"
                )
                return
            raise LifespanFailureError("startup", message.get("message", ""))
        elif message["type"] == "lifespan.shutdown.failed":
            self.shutdown.set()
            if (
                policy.current_policy(self.config).lifespan_mode
                is policy.LifespanMode.STRICT_SHUTDOWN
            ):
                run_phase = policy.current_phase("worker", create=True)
                assert run_phase is not None
                run_phase.shutdown_failed = True
                run_phase.clean_exit = False
                run_phase.observe(
                    "lifespan_mode", "shutdown_failure_strict", message.get("message", "")
                )
                await self.config.log.error(
                    "Lifespan shutdown failure, policy strict_shutdown prevents a clean worker "
                    f"exit: {message.get('message', '')}"
                )
            raise LifespanFailureError("shutdown", message.get("message", ""))
        else:
            raise UnexpectedMessageError(message["type"])
