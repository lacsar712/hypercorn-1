from __future__ import annotations

import pytest
import trio

from hypercorn.config import Config
from hypercorn.policy import (
    H2Surface,
    KeepPolicy,
    bind_phase,
    current_phase,
    current_policy,
    policy_frame,
)


def test_nested_pop_restores_outer_frame() -> None:
    # contextvars work identically outside trio tasks.
    with policy_frame(keep_policy=KeepPolicy.FORCE_CLOSE):
        assert current_policy(Config()).keep_policy is KeepPolicy.FORCE_CLOSE
        with policy_frame(keep_policy=KeepPolicy.DRAIN_THEN_CLOSE):
            assert current_policy(Config()).keep_policy is KeepPolicy.DRAIN_THEN_CLOSE
        assert current_policy(Config()).keep_policy is KeepPolicy.FORCE_CLOSE
    assert current_policy(Config()).keep_policy is KeepPolicy.HONOR_KEEPALIVE


@pytest.mark.trio
async def test_concurrent_trio_connections_do_not_share_a_stack() -> None:
    seen: dict[int, object] = {}

    async def connection(index: int, keep: KeepPolicy, surface: H2Surface) -> None:
        with bind_phase("connection"), policy_frame(keep_policy=keep, h2_surface=surface):
            await trio.sleep(0)
            effective = current_policy(Config())
            seen[index] = (effective.keep_policy, effective.h2_surface)
            await trio.sleep(0)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(connection, 0, KeepPolicy.FORCE_CLOSE, H2Surface.SINGLE_STREAM)
        nursery.start_soon(connection, 1, KeepPolicy.HONOR_KEEPALIVE, H2Surface.HONOR_MULTIPLEX)
        nursery.start_soon(connection, 2, KeepPolicy.DRAIN_THEN_CLOSE, H2Surface.DENY_PUSH)

    assert seen == {
        0: (KeepPolicy.FORCE_CLOSE, H2Surface.SINGLE_STREAM),
        1: (KeepPolicy.HONOR_KEEPALIVE, H2Surface.HONOR_MULTIPLEX),
        2: (KeepPolicy.DRAIN_THEN_CLOSE, H2Surface.DENY_PUSH),
    }
    # The per-connection bindings leave with the connection tasks.
    assert current_phase("connection") is None
    assert current_policy(Config()).keep_policy is KeepPolicy.HONOR_KEEPALIVE
