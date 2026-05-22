from __future__ import annotations

import gc

import numpy as np
import pytest

from arepy_ecs import Component, VectorValue, World


class Position(Component):
    x: float = 0.0
    y: float = 0.0


class Vec2(VectorValue):
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


class Transform(Component):
    position: Vec2 = Vec2()


class ExternalVec2(VectorValue):
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


class Body(Component):
    position: ExternalVec2 = ExternalVec2()


class Sprite(Component):
    position: Vec2 = Vec2()
    _payload: object | None = None


class Label(Component):
    name: str = ""


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


def test_vector_component_proxy_reuses_cached_vec_proxy_and_roundtrips_to_storage() -> None:
    world = World("demo")
    entity = world.create_entity().with_component(Transform(position=Vec2(1.0, 2.0))).build()

    transform = entity.get_component(Transform)
    first_position = transform.position
    second_position = transform.position

    assert first_position is second_position
    assert first_position.x == pytest.approx(1.0)
    assert first_position.y == pytest.approx(2.0)

    first_position.x = 9.5
    first_position.y = -4.0

    refreshed = entity.get_component(Transform)
    assert refreshed.position.x == pytest.approx(9.5)
    assert refreshed.position.y == pytest.approx(-4.0)


def test_custom_vectorvalue_subclass_is_detected_and_exposed_as_batch() -> None:
    world = World("demo")
    entity = world.create_entity().with_component(Body(position=ExternalVec2(2.0, 3.0))).build()

    body = entity.get_component(Body)
    assert isinstance(body.position, ExternalVec2)
    assert body.position.x == pytest.approx(2.0)
    assert body.position.y == pytest.approx(3.0)

    batch = world.get_registry().component_batch(Body)
    assert isinstance(batch.position, ExternalVec2)
    assert batch.position.x.dtype == np.float32
    assert batch.position.y.dtype == np.float32

    batch.position.x += np.array([5.0], dtype=np.float32)
    assert entity.get_component(Body).position.x == pytest.approx(7.0)


def test_private_component_fields_are_kept_on_python_side() -> None:
    world = World("demo")
    payload = object()
    entity = (
        world.create_entity()
        .with_component(Sprite(position=Vec2(3.0, 4.0), _payload=payload))
        .build()
    )

    sprite = entity.get_component(Sprite)

    assert sprite._payload is payload

    next_payload = object()
    sprite._payload = next_payload

    refreshed = entity.get_component(Sprite)
    assert refreshed._payload is next_payload


def test_string_component_fields_roundtrip_through_interned_storage() -> None:
    world = World("demo")
    entity = world.create_entity().with_component(Label(name="player")).build()

    label = entity.get_component(Label)
    assert label.name == "player"

    label.name = "enemy"

    refreshed = entity.get_component(Label)
    assert refreshed.name == "enemy"


def test_string_component_batch_returns_python_string_snapshots() -> None:
    world = World("demo")
    registry = world.get_registry()
    world.create_entity().with_component(Label(name="player")).build()
    world.create_entity().with_component(Label(name="enemy")).build()

    batch = registry.component_batch(Label)

    assert batch.name == ["player", "enemy"]


def test_string_component_fields_reject_ndarray_exports() -> None:
    world = World("demo")
    registry = world.get_registry()
    world.create_entity().with_component(Label(name="player")).build()

    with pytest.raises(TypeError, match="does not expose raw views"):
        registry.component_field_array(Label, "name")
