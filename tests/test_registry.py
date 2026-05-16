from __future__ import annotations

import gc

import numpy as np
import pytest

from arepy_ecs import Component, World


class Position(Component):
    x: float = 0.0
    y: float = 0.0


def test_component_proxy_mutations_roundtrip_to_rust_storage() -> None:
    world = World("demo")
    entity = world.create_entity().with_component(Position(x=1.0, y=2.0)).build()

    position = entity.get_component(Position)
    position.x = 9.5

    refreshed = entity.get_component(Position)
    assert refreshed.x == 9.5


def test_component_field_array_exposes_zero_copy_numpy_view() -> None:
    world = World("demo")
    registry = world.get_registry()
    first = world.create_entity().with_component(Position(x=1.25, y=2.5)).build()
    second = world.create_entity().with_component(Position(x=4.5, y=9.0)).build()

    xs = registry.component_field_array(Position, "x")

    assert isinstance(xs, np.ndarray)
    assert xs.dtype == np.float32
    assert xs.tolist() == [1.25, 4.5]

    xs += np.array([1.0, -2.0], dtype=np.float32)

    assert first.get_component(Position).x == 2.25
    assert second.get_component(Position).x == 2.5


def test_component_field_memoryview_is_writable_and_updates_rust_storage() -> None:
    world = World("demo")
    registry = world.get_registry()
    entity = world.create_entity().with_component(Position(x=1.0, y=2.0)).build()

    raw = registry.component_field_memoryview(Position, "x")

    assert isinstance(raw, memoryview)
    assert raw.readonly is False

    np.frombuffer(raw, dtype=np.float32)[0] = 8.75

    assert entity.get_component(Position).x == 8.75


def test_component_field_view_blocks_structural_mutation_until_released() -> None:
    world = World("demo")
    registry = world.get_registry()
    world.create_entity().with_component(Position(x=1.0, y=2.0)).build()

    view = registry.component_field_view(Position, "x")

    with pytest.raises(ValueError, match="active exported views"):
        world.create_entity().with_component(Position(x=3.0, y=4.0)).build()

    del view
    gc.collect()

    entity = world.create_entity().with_component(Position(x=3.0, y=4.0)).build()
    assert entity.get_component(Position).x == 3.0


def test_builtin_numeric_annotations_use_32_bit_storage() -> None:
    class Stats(Component):
        health: int = 100
        speed: float = 1.5

    world = World("demo")
    registry = world.get_registry()
    entity = world.create_entity().with_component(Stats(health=7, speed=2.5)).build()

    healths = registry.component_field_array(Stats, "health")
    speeds = registry.component_field_array(Stats, "speed")

    assert healths.dtype == np.int32
    assert speeds.dtype == np.float32
    assert int(healths[0]) == 7
    assert float(speeds[0]) == pytest.approx(2.5)
    assert entity.get_component(Stats).health == 7
