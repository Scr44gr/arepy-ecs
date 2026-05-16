from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import pytest

from arepy_ecs import Component, Entity, Query, With, World
from arepy_ecs.exceptions import ResourceNotFoundError
from arepy_ecs.systems import SystemPipeline, SystemState


class Position(Component):
    x: float = 0.0


@dataclass(slots=True)
class GameConfig:
    scale: float


@dataclass(slots=True)
class AudioConfig:
    volume: float


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


def test_world_add_systems_and_state_helpers_delegate_to_registry() -> None:
    world = World("demo")

    def update_system() -> None:
        return None

    def render_system() -> None:
        return None

    def input_system() -> None:
        return None

    world.add_systems(SystemPipeline.UPDATE, {update_system, input_system})
    world.add_system_with_state(SystemPipeline.RENDER, render_system, SystemState.OFF)
    world.set_system_state(SystemPipeline.UPDATE, input_system, SystemState.OFF)

    systems = world.get_registry().systems

    assert update_system in systems[SystemPipeline.UPDATE][SystemState.ON]
    assert input_system in systems[SystemPipeline.UPDATE][SystemState.OFF]
    assert render_system in systems[SystemPipeline.RENDER][SystemState.OFF]


def test_world_get_resource_falls_back_to_global_resources() -> None:
    global_config = GameConfig(scale=4.0)
    world = World("demo", global_resources={type(global_config).__name__: global_config})

    assert world.get_resource(GameConfig) is global_config
    assert world.get_global_resource(GameConfig) is global_config
    assert world.get_world_resource(GameConfig) is None


def test_world_local_resources_override_global_resources() -> None:
    world = World("demo", global_resources={"GameConfig": GameConfig(scale=2.0)})
    local_config = GameConfig(scale=6.0)
    world.add_resource(local_config)

    assert world.get_resource(GameConfig) is local_config
    assert world.get_world_resource(GameConfig) is local_config
    assert world.get_global_resource(GameConfig) is not None


def test_world_get_resource_supports_module_resources() -> None:
    module_resource = ModuleType("demo_module")
    world = World("demo")
    world.add_resource(module_resource)

    assert world.get_resource(module_resource) is module_resource
    assert world.get_world_resource(module_resource) is module_resource


def test_world_get_resource_raises_for_missing_resource() -> None:
    world = World("demo")

    with pytest.raises(ResourceNotFoundError, match="AudioConfig"):
        world.get_resource(AudioConfig)


def test_world_registers_all_lifecycle_callbacks() -> None:
    world = World("demo")

    @world.on_startup
    def startup() -> None:
        return None

    @world.on_update
    def update() -> None:
        return None

    @world.on_render
    def render() -> None:
        return None

    @world.on_shutdown
    def shutdown() -> None:
        return None

    assert world._startup_callbacks == [startup]
    assert world._update_callbacks == [update]
    assert world._render_callbacks == [render]
    assert world._shutdown_callbacks == [shutdown]
