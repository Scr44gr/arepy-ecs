from __future__ import annotations

import pytest

from arepy_ecs import Component, Entity, Query, With, Without, World
from arepy_ecs.systems import SystemPipeline


class Position(Component):
    x: float = 0.0
    y: float = 0.0


class Velocity(Component):
    x: float = 0.0
    y: float = 0.0


def movement_system(query: Query[Entity, With[Position, Velocity]]) -> None:
    for position, velocity in query.iter_components(Position, Velocity):
        position.x += velocity.x
        position.y += velocity.y


def test_registry_runs_query_backed_systems() -> None:
    world = World("demo")
    entity = (
        world.create_entity()
        .with_component(Position(x=1.0, y=2.0))
        .with_component(Velocity(x=0.5, y=-0.25))
        .build()
    )
    world.add_system(SystemPipeline.UPDATE, movement_system)

    world.get_registry().run(SystemPipeline.UPDATE)

    position = entity.get_component(Position)
    assert position.x == pytest.approx(1.5)
    assert position.y == pytest.approx(1.75)


def test_query_without_filter_excludes_entities() -> None:
    world = World("demo")
    moving = (
        world.create_entity()
        .with_component(Position(x=1.0, y=1.0))
        .with_component(Velocity(x=1.0, y=1.0))
        .build()
    )
    static_entity = world.create_entity().with_component(Position(x=4.0, y=5.0)).build()

    query: Query[Entity, tuple[With[Position], Without[Velocity]]] = Query(
        include=(Position,), exclude=(Velocity,)
    )
    query.set_registry(world.get_registry())

    assert moving not in query.get_entities()
    assert static_entity in query.get_entities()


def test_query_result_returns_batch_component_views() -> None:
    world = World("demo")
    query: Query[Entity, With[Position, Velocity]] = Query(include=(Position, Velocity))
    query.set_registry(world.get_registry())

    world.create_entity().with_component(Position(x=1.0, y=2.0)).with_component(
        Velocity(x=0.5, y=1.0)
    ).build()
    world.create_entity().with_component(Position(x=3.0, y=4.0)).with_component(
        Velocity(x=-1.0, y=2.0)
    ).build()

    position, velocity = query.result(Position, Velocity)

    position.x += velocity.x
    position.y += velocity.y

    entities = world.get_registry().query_entities((Position, Velocity), ())
    assert entities[0].get_component(Position).x == pytest.approx(1.5)
    assert entities[0].get_component(Position).y == pytest.approx(3.0)
    assert entities[1].get_component(Position).x == pytest.approx(2.0)
    assert entities[1].get_component(Position).y == pytest.approx(6.0)
