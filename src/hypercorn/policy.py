"""Service run policy stack.

A stack of :class:`PolicyFrame` instances bound to the run currently being
worked on by a worker. A run is either the worker level lifespan run or a
single connection HTTP/WebSocket session.

The stack is deliberately decoupled from :class:`~hypercorn.config.Config`
flat fields, from the h11/h2 state machines and from StatsD counters. It is
transported in :mod:`contextvars` so that:

* there is no process level singleton, the value is owned by the context of
  the task working on a run;
* both the asyncio and trio backends can push frames without any backend
  specific code in this module;
* two concurrent connections are separate tasks with separate context
  copies - a frame pushed for one connection can never leak onto another.

Frames are pushed for a limited scope and popped on exit (including via an
exception). The empty stack reproduces the existing behaviour exactly.

Four independent dimensions are supported, each an enum rather than a
collapsed boolean. The read only effective view is resolved innermost frame
wins and merged with the Config baseline and a snapshot of the current run
phase, see :func:`current_policy`.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


class LifespanMode(str, Enum):
    HONOR_STARTUP = "honor_startup"
    SUPPRESS_STARTUP = "suppress_startup"
    STRICT_SHUTDOWN = "strict_shutdown"


class KeepPolicy(str, Enum):
    HONOR_KEEPALIVE = "honor_keepalive"
    DRAIN_THEN_CLOSE = "drain_then_close"
    FORCE_CLOSE = "force_close"


class H2Surface(str, Enum):
    HONOR_MULTIPLEX = "honor_multiplex"
    DENY_PUSH = "deny_push"
    SINGLE_STREAM = "single_stream"


class WSMode(str, Enum):
    HONOR_WS = "honor_ws"
    DENY_UPGRADE = "deny_upgrade"
    CLOSE_ON_EXC = "close_on_exc"


class PolicyTimingError(RuntimeError):
    """Raised when a frame is pushed after the exit it targets has frozen.

    An exit is frozen once its decision has been made, for example once a
    response has started finalising. A late push is rejected rather than
    being silently applied to a snapshot taken on entry.
    """


@dataclass(frozen=True)
class PolicyFrame:
    """One immutable frame of policy overrides for a run scope."""

    lifespan_mode: LifespanMode = LifespanMode.HONOR_STARTUP
    keep_policy: KeepPolicy = KeepPolicy.HONOR_KEEPALIVE
    h2_surface: H2Surface = H2Surface.HONOR_MULTIPLEX
    ws_mode: WSMode = WSMode.HONOR_WS


@dataclass(frozen=True)
class PolicyObservation:
    """An observable policy decision taken at an exit."""

    dimension: str
    action: str
    detail: str = ""


@dataclass(frozen=True)
class PhaseSnapshot:
    """Read only snapshot of the run phase merged into the effective view."""

    startup_decided: bool = False
    shutdown_failed: bool = False
    clean_exit: bool = True
    response_finalizing: bool = False
    multiplex_opened: bool = False
    ws_accepted: bool = False


class RunPhase:
    """Observable lifecycle state of the run the stack is bound to.

    A worker phase tracks lifespan state, a connection phase tracks the
    state of a single HTTP/WebSocket connection. Instances are not shared
    between concurrent connections - each connection binds its own phase in
    its own context.
    """

    def __init__(self, scope: str) -> None:
        self.scope = scope
        # Lifespan (worker phase)
        self.startup_decided = False
        self.shutdown_failed = False
        self.clean_exit = True
        # Connection finalisation (connection phase)
        self.response_finalizing = False
        # h2 multiplexing (connection phase)
        self.multiplex_opened = False
        # WebSocket handshake (connection phase)
        self.ws_accepted = False
        self.observations: list[PolicyObservation] = []

    def observe(self, dimension: str, action: str, detail: str = "") -> PolicyObservation:
        observation = PolicyObservation(dimension, action, detail)
        self.observations.append(observation)
        return observation

    def snapshot(self) -> PhaseSnapshot:
        return PhaseSnapshot(
            startup_decided=self.startup_decided,
            shutdown_failed=self.shutdown_failed,
            clean_exit=self.clean_exit,
            response_finalizing=self.response_finalizing,
            multiplex_opened=self.multiplex_opened,
            ws_accepted=self.ws_accepted,
        )


@dataclass(frozen=True)
class EffectivePolicy:
    """Read only view of the frames merged with the Config baseline."""

    lifespan_mode: LifespanMode
    keep_policy: KeepPolicy
    h2_surface: H2Surface
    ws_mode: WSMode
    keep_alive_timeout: float
    h2_max_concurrent_streams: int
    websocket_ping_interval: float | None
    phase: PhaseSnapshot = field(default_factory=PhaseSnapshot)


_frames: ContextVar[tuple[PolicyFrame, ...]] = ContextVar("hypercorn_policy_frames", default=())
_worker_phase: ContextVar[RunPhase | None] = ContextVar("hypercorn_worker_phase", default=None)
_connection_phase: ContextVar[RunPhase | None] = ContextVar(
    "hypercorn_connection_phase", default=None
)


def current_frames() -> tuple[PolicyFrame, ...]:
    """Return the active frames, outermost first."""
    return _frames.get()


def current_phase(scope: str, *, create: bool = False) -> RunPhase | None:
    """Return the phase bound to the current context for ``scope``.

    With ``create=True`` a phase is bound to the current context if none is
    present, allowing protocol objects driven directly (e.g. in tests) to
    record lifecycle state without a worker entry point.
    """
    variable = _worker_phase if scope == "worker" else _connection_phase
    phase = variable.get()
    if phase is None and create:
        phase = RunPhase(scope)
        variable.set(phase)
    return phase


def ensure_phase(scope: str) -> RunPhase:
    """Bind a phase for ``scope`` to the current context if needed.

    The binding lives for the remainder of the current context, which for
    the worker and per-connection tasks is the lifetime of the run. Use
    :class:`bind_phase` where a scoped, reset binding is required.
    """
    phase = current_phase(scope, create=True)
    assert phase is not None  # noqa: S101
    return phase


class bind_phase(AbstractContextManager["bind_phase"]):
    """Bind a fresh run phase, resetting the previous binding on exit."""

    def __init__(self, scope: str) -> None:
        self.scope = scope
        self.phase = RunPhase(scope)
        self._token: Token[RunPhase | None] | None = None

    def __enter__(self) -> bind_phase:
        variable = _worker_phase if self.scope == "worker" else _connection_phase
        self._token = variable.set(self.phase)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._token is not None:
            variable = _worker_phase if self.scope == "worker" else _connection_phase
            variable.reset(self._token)
            self._token = None

    async def __aenter__(self) -> bind_phase:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_value, traceback)


def _merged_phase_snapshot() -> PhaseSnapshot:
    worker = _worker_phase.get()
    connection = _connection_phase.get()
    phases = [phase for phase in (worker, connection) if phase is not None]
    if not phases:
        return PhaseSnapshot()
    return PhaseSnapshot(
        startup_decided=any(phase.startup_decided for phase in phases),
        shutdown_failed=any(phase.shutdown_failed for phase in phases),
        clean_exit=all(phase.clean_exit for phase in phases),
        response_finalizing=any(phase.response_finalizing for phase in phases),
        multiplex_opened=any(phase.multiplex_opened for phase in phases),
        ws_accepted=any(phase.ws_accepted for phase in phases),
    )


def current_policy(config: Config) -> EffectivePolicy:
    """Resolve the read only effective policy for the current run.

    Frames are merged innermost wins over the Config baseline. Frames with
    honor values do not override outer frames. This never mutates Config.
    """
    lifespan_mode = LifespanMode.HONOR_STARTUP
    keep_policy = KeepPolicy.HONOR_KEEPALIVE
    h2_surface = H2Surface.HONOR_MULTIPLEX
    ws_mode = WSMode.HONOR_WS
    for frame in _frames.get():
        if frame.lifespan_mode is not LifespanMode.HONOR_STARTUP:
            lifespan_mode = frame.lifespan_mode
        if frame.keep_policy is not KeepPolicy.HONOR_KEEPALIVE:
            keep_policy = frame.keep_policy
        if frame.h2_surface is not H2Surface.HONOR_MULTIPLEX:
            h2_surface = frame.h2_surface
        if frame.ws_mode is not WSMode.HONOR_WS:
            ws_mode = frame.ws_mode
    return EffectivePolicy(
        lifespan_mode=lifespan_mode,
        keep_policy=keep_policy,
        h2_surface=h2_surface,
        ws_mode=ws_mode,
        keep_alive_timeout=config.keep_alive_timeout,
        h2_max_concurrent_streams=config.h2_max_concurrent_streams,
        websocket_ping_interval=config.websocket_ping_interval,
        phase=_merged_phase_snapshot(),
    )


def push(frame: PolicyFrame) -> Token[tuple[PolicyFrame, ...]]:
    """Push a frame onto the stack for the current context.

    Raises :class:`PolicyTimingError` if the frame attempts to alter an exit
    whose decision has already frozen. Pushing an honor value is always
    allowed as it does not alter any exit.
    """
    if frame.lifespan_mode is LifespanMode.SUPPRESS_STARTUP:
        phase = _worker_phase.get()
        if phase is not None and phase.startup_decided:
            raise PolicyTimingError(
                "Cannot push suppress_startup after lifespan startup failure " "has been decided"
            )
    if frame.keep_policy is not KeepPolicy.HONOR_KEEPALIVE:
        phase = _connection_phase.get()
        if phase is not None and phase.response_finalizing:
            raise PolicyTimingError(
                f"Cannot push {frame.keep_policy.value} after the response has "
                "started finalising the connection"
            )
    if frame.h2_surface is H2Surface.SINGLE_STREAM:
        phase = _connection_phase.get()
        if phase is not None and phase.multiplex_opened:
            raise PolicyTimingError(
                "Cannot push single_stream after the h2 connection has opened "
                "multiple concurrent streams"
            )
    if frame.ws_mode is WSMode.DENY_UPGRADE:
        phase = _connection_phase.get()
        if phase is not None and phase.ws_accepted:
            raise PolicyTimingError(
                "Cannot push deny_upgrade after the WebSocket handshake has " "been accepted"
            )
    return _frames.set(_frames.get() + (frame,))


def pop(token: Token[tuple[PolicyFrame, ...]]) -> None:
    """Pop a frame previously pushed in the current context."""
    _frames.reset(token)


class policy_frame(AbstractContextManager["policy_frame"]):
    """Context manager pushing a frame for the duration of a scope.

    The frame is popped on exit including when the scope raises, so exception
    paths pop cleanly and the outer frame becomes active again.

    Either a :class:`PolicyFrame` is given or dimensions are passed as
    keyword arguments.
    """

    def __init__(
        self,
        frame: PolicyFrame | None = None,
        *,
        lifespan_mode: LifespanMode = LifespanMode.HONOR_STARTUP,
        keep_policy: KeepPolicy = KeepPolicy.HONOR_KEEPALIVE,
        h2_surface: H2Surface = H2Surface.HONOR_MULTIPLEX,
        ws_mode: WSMode = WSMode.HONOR_WS,
    ) -> None:
        self._frame = (
            frame
            if frame is not None
            else PolicyFrame(
                lifespan_mode=lifespan_mode,
                keep_policy=keep_policy,
                h2_surface=h2_surface,
                ws_mode=ws_mode,
            )
        )
        self._token: Token[tuple[PolicyFrame, ...]] | None = None

    def __enter__(self) -> policy_frame:
        self._token = push(self._frame)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        token = self._token
        if token is not None:
            pop(token)
            self._token = None

    # The contextvar mechanics are synchronous, allow use within async
    # code via ``async with`` as well.
    async def __aenter__(self) -> policy_frame:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_value, traceback)


def observations(scope: str = "connection") -> tuple[PolicyObservation, ...]:
    """Return observations recorded on the phase bound to the context."""
    phase = current_phase(scope)
    if phase is None:
        return ()
    return tuple(phase.observations)


__all__ = (
    "bind_phase",
    "EffectivePolicy",
    "H2Surface",
    "KeepPolicy",
    "LifespanMode",
    "PhaseSnapshot",
    "PolicyFrame",
    "PolicyObservation",
    "PolicyTimingError",
    "RunPhase",
    "WSMode",
    "current_frames",
    "current_phase",
    "current_policy",
    "ensure_phase",
    "observations",
    "policy_frame",
    "pop",
    "push",
)
