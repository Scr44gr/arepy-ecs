from __future__ import annotations

import pytest

from arepy_ecs import Component, EntityBuilder, Registry, World


class Position(Component):
    x: float = 0.0
    y: float = 0.0


class Velocity(Component):
    x: float = 0.0
    y: float = 0.0


class Health(Component):
    value: int = 100


@pytest.fixture
def registry() -> Registry:
    return Registry()


@pytest.fixture
def entity_builder(registry: Registry) -> EntityBuilder:
    return EntityBuilder(registry.create_entity(), registry)


def test_entity_builder_creation(registry: Registry) -> None:
    entity = registry.create_entity()
    builder = EntityBuilder(entity, registry)

    assert builder._entity is entity
    assert builder._registry is registry
    assert builder._components == []


def test_entity_builder_with_component_tracks_components_and_chains(
    entity_builder: EntityBuilder,
) -> None:
    position = Position(x=10.0, y=20.0)
    velocity = Velocity(x=5.0, y=3.0)

    builder = entity_builder.with_component(position).with_component(velocity)

    assert builder is entity_builder
    assert entity_builder._components == [position, velocity]


def test_entity_builder_invalid_component_raises_type_error(
    entity_builder: EntityBuilder,
) -> None:
    with pytest.raises(TypeError, match="Component must be of type Component"):
        entity_builder.with_component("not_a_component")  # type: ignore[arg-type]


def test_entity_builder_duplicate_component_raises_type_error(
    entity_builder: EntityBuilder,
) -> None:
    entity_builder.with_component(Position(x=1.0, y=2.0))

    with pytest.raises(TypeError, match="already exists in entity"):
        entity_builder.with_component(Position(x=3.0, y=4.0))


def test_entity_builder_build_persists_components(entity_builder: EntityBuilder) -> None:
    built_entity = (
        entity_builder.with_component(Position(x=15.0, y=25.0))
        .with_component(Velocity(x=7.0, y=9.0))
        .with_component(Health(value=150))
        .build()
    )

    assert built_entity is entity_builder._entity
    assert built_entity.get_component(Position).x == pytest.approx(15.0)
    assert built_entity.get_component(Position).y == pytest.approx(25.0)
    assert built_entity.get_component(Velocity).x == pytest.approx(7.0)
    assert built_entity.get_component(Velocity).y == pytest.approx(9.0)
    assert built_entity.get_component(Health).value == 150


def test_entity_builder_empty_build(entity_builder: EntityBuilder) -> None:
    built_entity = entity_builder.build()

    assert built_entity is entity_builder._entity
    assert entity_builder._components == []


def test_entity_builder_fluent_interface(registry: Registry) -> None:
    entity = registry.create_entity()

    built_entity = (
        EntityBuilder(entity, registry)
        .with_component(Position(x=1.0, y=2.0))
        .with_component(Velocity(x=3.0, y=4.0))
        .with_component(Health(value=75))
        .build()
    )

    assert built_entity is entity
    assert registry.has_component(entity, Position)
    assert registry.has_component(entity, Velocity)
    assert registry.has_component(entity, Health)


def test_world_entity_builder_integration() -> None:
    world = World("integration")

    entity = (
        world.create_entity()
        .with_component(Position(x=100.0, y=200.0))
        .with_component(Velocity(x=10.0, y=20.0))
        .build()
    )

    assert entity.get_component(Position).x == pytest.approx(100.0)
    assert entity.get_component(Position).y == pytest.approx(200.0)
    assert entity.get_component(Velocity).x == pytest.approx(10.0)
    assert entity.get_component(Velocity).y == pytest.approx(20.0)
