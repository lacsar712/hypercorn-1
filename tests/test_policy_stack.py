from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from hypercorn.config import Config
from hypercorn import policy
from hypercorn.policy import (
    H2Surface,
    KeepPolicy,
    LifespanMode,
    PolicyFrame,
    PolicyTimingError,
    WSMode,
    bind_phase,
    current_policy,
    policy_frame,
    push,
)


def test_empty_stack_matches_config_baseline() -> None:
    config = Config()
    config.keep_alive_timeout = 12.5
    config.h2_max_concurrent_streams = 7
    config.websocket_ping_interval = 3.0
    effective = current_policy(config)
    assert effective.lifespan_mode is LifespanMode.HONOR_STARTUP
    assert effective.keep_policy is KeepPolicy.HONOR_KEEPALIVE
    assert effective.h2_surface is H2Surface.HONOR_MULTIPLEX
    assert effective.ws_mode is WSMode.HONOR_WS
    assert effective.keep_alive_timeout == 12.5
    assert effective.h2_max_concurrent_streams == 7
    assert effective.websocket_ping_interval == 3.0


def test_effective_view_is_read_only() -> None:
    with pytest.raises(FrozenInstanceError):
        current_policy(Config()).keep_policy = KeepPolicy.FORCE_CLOSE  # type: ignore[misc]


def test_config_is_not_mutated_by_a_frame() -> None:
    config = Config()
    config.keep_alive_timeout = 9.0
    with policy_frame(PolicyFrame(keep_policy=KeepPolicy.FORCE_CLOSE)):
        effective = current_policy(config)
        assert effective.keep_policy is KeepPolicy.FORCE_CLOSE
        assert config.keep_alive_timeout == 9.0
    assert config.keep_alive_timeout == 9.0
    assert current_policy(config).keep_policy is KeepPolicy.HONOR_KEEPALIVE


def test_dimensions_are_independent_enums() -> None:
    with policy_frame(PolicyFrame(lifespan_mode=LifespanMode.STRICT_SHUTDOWN)):
        effective = current_policy(Config())
        assert effective.lifespan_mode is LifespanMode.STRICT_SHUTDOWN
        assert effective.keep_policy is KeepPolicy.HONOR_KEEPALIVE
        assert effective.h2_surface is H2Surface.HONOR_MULTIPLEX
        assert effective.ws_mode is WSMode.HONOR_WS


def test_inner_pop_returns_to_outer_frame() -> None:
    outer = PolicyFrame(keep_policy=KeepPolicy.FORCE_CLOSE, ws_mode=WSMode.DENY_UPGRADE)
    inner = PolicyFrame(keep_policy=KeepPolicy.DRAIN_THEN_CLOSE)
    with policy_frame(outer):
        assert current_policy(Config()).keep_policy is KeepPolicy.FORCE_CLOSE
        with policy_frame(inner):
            effective = current_policy(Config())
            assert effective.keep_policy is KeepPolicy.DRAIN_THEN_CLOSE
            # The outer frame still governs untouched dimensions.
            assert effective.ws_mode is WSMode.DENY_UPGRADE
        # Inner pop returns to the outer frame.
        assert current_policy(Config()).keep_policy is KeepPolicy.FORCE_CLOSE
        assert current_policy(Config()).ws_mode is WSMode.DENY_UPGRADE
    assert current_policy(Config()).keep_policy is KeepPolicy.HONOR_KEEPALIVE


def test_exception_path_pops_clean() -> None:
    with pytest.raises(RuntimeError):
        with policy_frame(keep_policy=KeepPolicy.FORCE_CLOSE):
            raise RuntimeError("boom")
    assert current_policy(Config()).keep_policy is KeepPolicy.HONOR_KEEPALIVE


def test_phase_snapshot_merged_into_view() -> None:
    with bind_phase("connection") as bound:
        bound.phase.response_finalizing = True
        bound.phase.multiplex_opened = True
        effective = current_policy(Config())
        assert effective.phase.response_finalizing is True
        assert effective.phase.multiplex_opened is True
        assert effective.phase.ws_accepted is False
    # The phase binding is reset on exit.
    assert current_policy(Config()).phase.response_finalizing is False


def test_late_push_suppress_startup_rejected() -> None:
    with bind_phase("worker") as bound:
        bound.phase.startup_decided = True
        with pytest.raises(PolicyTimingError):
            push(PolicyFrame(lifespan_mode=LifespanMode.SUPPRESS_STARTUP))


def test_late_push_force_close_rejected() -> None:
    with bind_phase("connection") as bound:
        bound.phase.response_finalizing = True
        with pytest.raises(PolicyTimingError):
            push(PolicyFrame(keep_policy=KeepPolicy.FORCE_CLOSE))


def test_late_push_drain_then_close_rejected() -> None:
    with bind_phase("connection") as bound:
        bound.phase.response_finalizing = True
        with pytest.raises(PolicyTimingError):
            push(PolicyFrame(keep_policy=KeepPolicy.DRAIN_THEN_CLOSE))


def test_late_push_single_stream_rejected() -> None:
    with bind_phase("connection") as bound:
        bound.phase.multiplex_opened = True
        with pytest.raises(PolicyTimingError):
            push(PolicyFrame(h2_surface=H2Surface.SINGLE_STREAM))


def test_late_push_deny_upgrade_rejected() -> None:
    with bind_phase("connection") as bound:
        bound.phase.ws_accepted = True
        with pytest.raises(PolicyTimingError):
            push(PolicyFrame(ws_mode=WSMode.DENY_UPGRADE))


def test_other_dimensions_still_pushable_after_a_phase_freezes() -> None:
    with bind_phase("connection") as bound:
        bound.phase.ws_accepted = True
        # deny_upgrade is frozen but a keep policy frame is unrelated.
        token = push(PolicyFrame(keep_policy=KeepPolicy.FORCE_CLOSE))
        policy.pop(token)


@pytest.mark.asyncio
async def test_concurrent_connections_do_not_share_a_stack() -> None:
    seen: dict[int, KeepPolicy] = {}

    async def connection(index: int, keep: KeepPolicy) -> None:
        with policy_frame(keep_policy=keep):
            await asyncio.sleep(0)
            seen[index] = current_policy(Config()).keep_policy
            await asyncio.sleep(0)

    await asyncio.gather(
        connection(0, KeepPolicy.FORCE_CLOSE),
        connection(1, KeepPolicy.HONOR_KEEPALIVE),
        connection(2, KeepPolicy.DRAIN_THEN_CLOSE),
    )
    assert seen == {
        0: KeepPolicy.FORCE_CLOSE,
        1: KeepPolicy.HONOR_KEEPALIVE,
        2: KeepPolicy.DRAIN_THEN_CLOSE,
    }
    # The stacks are gone with the connection tasks.
    assert current_policy(Config()).keep_policy is KeepPolicy.HONOR_KEEPALIVE
