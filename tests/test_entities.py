from __future__ import annotations

import pytest

from arepy_ecs import Component, Entity, Registry
from arepy_ecs.exceptions import ComponentNotFoundError


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
def entity(registry: Registry) -> Entity:
    return registry.create_entity()


def test_entity_creation_and_id_assignment(registry: Registry) -> None:
    entity_1 = registry.create_entity()
    entity_2 = registry.create_entity()
    entity_3 = registry.create_entity()

    assert entity_2.get_id() == entity_1.get_id() + 1
    assert entity_3.get_id() == entity_2.get_id() + 1


def test_entity_component_operations(entity: Entity) -> None:
    entity.add_component(Position(x=10.0, y=20.0))
    entity.add_component(Velocity(x=5.0, y=3.0))

    assert entity.has_component(Position)
    assert entity.has_component(Velocity)
    assert entity.has_component(Health) is False

    position = entity.get_component(Position)
    velocity = entity.get_component(Velocity)

    assert position.x == pytest.approx(10.0)
    assert position.y == pytest.approx(20.0)
    assert velocity.x == pytest.approx(5.0)
    assert velocity.y == pytest.approx(3.0)

    entity.remove_component(Position)

    assert entity.has_component(Position) is False
    assert entity.has_component(Velocity)


def test_entity_component_cache_reuses_proxy_and_invalidates_on_remove(entity: Entity) -> None:
    entity.add_component(Position(x=15.0, y=25.0))

    first = entity.get_component(Position)
    second = entity.get_component(Position)

    assert first is second
    assert Position in entity._component_cache

    entity.remove_component(Position)

    assert Position not in entity._component_cache


def test_entity_component_not_found_raises(entity: Entity) -> None:
    with pytest.raises(ComponentNotFoundError, match="Position"):
        entity.get_component(Position)


def test_entity_equality_hash_and_string_representation(registry: Registry) -> None:
    entity_1 = registry.create_entity()
    entity_2 = registry.create_entity()
    entity_3 = Entity(registry, entity_1.get_id())

    assert entity_1 == entity_3
    assert entity_1 != entity_2
    assert hash(entity_1) == hash(entity_3)
    assert hash(entity_1) != hash(entity_2)
    assert str(entity_1) == f"Entity(id={entity_1.get_id()})"
    assert repr(entity_1) == str(entity_1)


def test_entity_kill_clears_component_cache(entity: Entity, registry: Registry) -> None:
    entity.add_component(Position(x=1.0, y=2.0))
    _ = entity.get_component(Position)

    entity.kill()

    assert entity._component_cache == {}
    assert registry.number_of_entities == 0


def test_entity_works_in_sets_and_dicts(registry: Registry) -> None:
    entity_1 = registry.create_entity()
    entity_2 = registry.create_entity()
    entity_3 = Entity(registry, entity_1.get_id())

    entity_set = {entity_1, entity_2, entity_3}
    entity_map = {entity_1: "first", entity_2: "second"}

    assert len(entity_set) == 2
    assert entity_map[entity_3] == "first"
