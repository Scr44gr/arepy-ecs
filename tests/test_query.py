from __future__ import annotations

import numpy as np
import pytest

from arepy_ecs import Component, Entity, Query, VectorValue, With, Without, World
from arepy_ecs.query import (
    QueryDefinitionError,
    build_query_from_annotation,
    get_queries_instance_from_arguments,
    get_signed_query_arguments,
    sign_queries,
)
from arepy_ecs.systems import SystemPipeline


class Vec2(VectorValue):
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


class Position(Component):
    x: float = 0.0
    y: float = 0.0


class Velocity(Component):
    x: float = 0.0
    y: float = 0.0


class Transform(Component):
    position: Vec2 = Vec2()


class Motion(Component):
    velocity: Vec2 = Vec2()


class NameTag(Component):
    name: str = ""


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


def test_query_get_batch_returns_logical_vector_batches() -> None:
    world = World("demo")
    query: Query[Entity, With[Transform, Motion]] = Query(include=(Transform, Motion))
    query.set_registry(world.get_registry())

    world.create_entity().with_component(Transform(position=Vec2(1.0, 2.0))).with_component(
        Motion(velocity=Vec2(0.5, -1.0))
    ).build()
    world.create_entity().with_component(Transform(position=Vec2(3.0, 4.0))).with_component(
        Motion(velocity=Vec2(-2.0, 1.5))
    ).build()

    positions = query.get_batch(Transform, "position")
    velocities = query.get_batch(Motion, "velocity")

    assert isinstance(positions, Vec2)
    assert isinstance(positions.x, np.ndarray)
    assert positions.x.dtype == np.float32
    assert positions.y.dtype == np.float32

    positions.x += velocities.x
    positions.y += velocities.y

    entities = world.get_registry().query_entities((Transform, Motion), ())
    assert entities[0].get_component(Transform).position.x == pytest.approx(1.5)
    assert entities[0].get_component(Transform).position.y == pytest.approx(1.0)
    assert entities[1].get_component(Transform).position.x == pytest.approx(1.0)
    assert entities[1].get_component(Transform).position.y == pytest.approx(5.5)


def test_query_get_batch_returns_scalar_field_arrays() -> None:
    world = World("demo")
    query: Query[Entity, With[Position]] = Query(include=(Position,))
    query.set_registry(world.get_registry())

    world.create_entity().with_component(Position(x=1.0, y=2.0)).build()
    world.create_entity().with_component(Position(x=3.5, y=4.5)).build()

    xs, ys = query.get_batch(Position, "x", "y")

    assert isinstance(xs, np.ndarray)
    assert isinstance(ys, np.ndarray)
    assert xs.dtype == np.float32
    assert ys.dtype == np.float32
    assert xs.tolist() == [1.0, 3.5]
    assert ys.tolist() == [2.0, 4.5]


def test_query_get_batch_returns_string_field_snapshots() -> None:
    world = World("demo")
    query: Query[Entity, With[NameTag]] = Query(include=(NameTag,))
    query.set_registry(world.get_registry())

    world.create_entity().with_component(NameTag(name="player")).build()
    world.create_entity().with_component(NameTag(name="enemy")).build()

    name_tags = query.get_batch(NameTag)

    assert name_tags.name == ["player", "enemy"]


def test_query_requires_registry_before_iteration() -> None:
    query: Query[Entity, With[Position]] = Query(include=(Position,))

    with pytest.raises(RuntimeError, match="not attached to a Registry"):
        list(query)

    with pytest.raises(RuntimeError, match="not attached to a Registry"):
        query.get_registry()


def test_query_iter_entities_components_returns_entity_and_components() -> None:
    world = World("demo")
    entity = (
        world.create_entity()
        .with_component(Position(x=5.0, y=6.0))
        .with_component(Velocity(x=7.0, y=8.0))
        .build()
    )
    query: Query[Entity, With[Position, Velocity]] = Query(include=(Position, Velocity))
    query.set_registry(world.get_registry())

    rows = list(query.iter_entities_components(Position, Velocity))

    assert len(rows) == 1
    resolved_entity, position, velocity = rows[0]
    assert resolved_entity is entity
    assert position.x == pytest.approx(5.0)
    assert position.y == pytest.approx(6.0)
    assert velocity.x == pytest.approx(7.0)
    assert velocity.y == pytest.approx(8.0)


def test_query_iter_components_updates_vector_components() -> None:
    world = World("demo")
    entity = (
        world.create_entity()
        .with_component(Transform(position=Vec2(2.0, 3.0)))
        .with_component(Motion(velocity=Vec2(0.5, -1.0)))
        .build()
    )
    query: Query[Entity, With[Transform, Motion]] = Query(include=(Transform, Motion))
    query.set_registry(world.get_registry())

    for transform, motion in query.iter_components(Transform, Motion):
        transform.position.x += motion.velocity.x
        transform.position.y += motion.velocity.y

    refreshed = entity.get_component(Transform)
    assert refreshed.position.x == pytest.approx(2.5)
    assert refreshed.position.y == pytest.approx(2.0)


def test_query_iter_components_updates_string_fields() -> None:
    world = World("demo")
    entity = world.create_entity().with_component(NameTag(name="player")).build()
    query: Query[Entity, With[NameTag]] = Query(include=(NameTag,))
    query.set_registry(world.get_registry())

    for (name_tag,) in query.iter_components(NameTag):
        name_tag.name = "enemy"

    assert entity.get_component(NameTag).name == "enemy"


def test_query_iter_components_restores_entity_proxy_behavior_after_iteration() -> None:
    world = World("demo")
    entity = world.create_entity().with_component(Position(x=1.0, y=2.0)).build()
    query: Query[Entity, With[Position]] = Query(include=(Position,))
    query.set_registry(world.get_registry())

    (position,) = next(query.iter_components(Position))
    position.x = 3.5

    assert entity.get_component(Position).x == pytest.approx(3.5)


def test_query_iter_components_cache_rebuilds_when_matching_entities_change() -> None:
    world = World("demo")
    world.create_entity().with_component(Position(x=1.0, y=2.0)).build()
    query: Query[Entity, With[Position]] = Query(include=(Position,))
    query.set_registry(world.get_registry())

    assert len(list(query.iter_components(Position))) == 1

    world.create_entity().with_component(Position(x=3.0, y=4.0)).build()

    rows = list(query.iter_components(Position))
    assert len(rows) == 2


def test_query_iteration_returns_empty_when_required_component_is_missing() -> None:
    world = World("demo")
    query: Query[Entity, With[Position, Velocity]] = Query(include=(Position, Velocity))
    query.set_registry(world.get_registry())

    world.create_entity().with_component(Position(x=1.0, y=2.0)).build()

    assert list(query.iter_components(Position, Velocity)) == []
    assert list(query.iter_entities_components(Position, Velocity)) == []
    assert query.get_entities() == set()


def test_query_without_syncs_on_component_add_and_remove() -> None:
    world = World("demo")
    entity = world.create_entity().build()
    query: Query[Entity, Without[Velocity]] = Query(exclude=(Velocity,))
    query.set_registry(world.get_registry())

    assert entity in query.get_entities()

    entity.add_component(Velocity(x=3.0, y=4.0))
    assert entity not in query.get_entities()

    entity.remove_component(Velocity)
    assert entity in query.get_entities()


def test_query_matches_respects_with_and_without_filters() -> None:
    query: Query[Entity, tuple[With[Position], Without[Velocity]]] = Query(
        include=(Position,),
        exclude=(Velocity,),
    )

    assert query.matches(("Position",)) is True
    assert query.matches(("Position", "Health")) is True
    assert query.matches(("Position", "Velocity")) is False
    assert query.matches(("Velocity",)) is False


def test_build_query_from_annotation_supports_tuple_filters() -> None:
    query = build_query_from_annotation(Query[Entity, tuple[With[Position], Without[Velocity]]])

    assert query.matches(("Position",)) is True
    assert query.matches(("Position", "Velocity")) is False


def test_build_query_from_annotation_rejects_unsupported_filters() -> None:
    with pytest.raises(QueryDefinitionError, match="Unsupported query filter annotation"):
        build_query_from_annotation(Query[Entity, Position])


def test_get_signed_query_arguments_reads_future_annotations() -> None:
    def complex_system(
        moving: Query[Entity, With[Position, Velocity]],
        static: Query[Entity, Without[Velocity]],
    ) -> None:
        return None

    arguments = get_signed_query_arguments(complex_system)

    assert tuple(arguments) == ("moving", "static")
    assert arguments["moving"] == Query[Entity, With[Position, Velocity]]
    assert arguments["static"] == Query[Entity, Without[Velocity]]


def test_sign_queries_builds_query_instances_from_annotations() -> None:
    signed_arguments = [
        ("moving", Query[Entity, With[Position, Velocity]]),
        ("static", Query[Entity, Without[Velocity]]),
    ]

    queries = sign_queries(signed_arguments)

    assert len(queries) == 2
    assert queries[0][0] == "moving"
    assert isinstance(queries[0][1], Query)
    assert queries[1][0] == "static"
    assert isinstance(queries[1][1], Query)


def test_get_queries_instance_from_arguments_filters_non_query_values() -> None:
    moving = Query[Entity, With[Position]](include=(Position,))
    static = Query[Entity, Without[Velocity]](exclude=(Velocity,))

    queries = get_queries_instance_from_arguments([moving, 123, "skip", static])

    assert queries == [moving, static]
