from __future__ import annotations

from dataclasses import dataclass

from arepy_ecs import Component, Entity, Query, With, World
from arepy_ecs.systems import SystemPipeline


class Position(Component):
    x: float = 0.0


@dataclass(slots=True)
class GameConfig:
    scale: float


def test_world_injects_resources_into_systems() -> None:
    world = World("demo")
    world.add_resource(GameConfig(scale=3.0))
    world.create_entity().with_component(Position(x=2.0)).build()

    def scale_system(query: Query[Entity, With[Position]], config: GameConfig) -> None:
        for (position,) in query.iter_components(Position):
            position.x *= config.scale

    world.add_system(SystemPipeline.UPDATE, scale_system)

    world.get_registry().run(SystemPipeline.UPDATE)

    entity = next(iter(world.get_registry().query_entities((Position,), ())))
    assert entity.get_component(Position).x == 6.0


def test_world_callback_registration_preserves_function_identity() -> None:
    world = World("demo")

    @world.on_startup
    def startup() -> None:
        return None

    assert startup.__name__ == "startup"
