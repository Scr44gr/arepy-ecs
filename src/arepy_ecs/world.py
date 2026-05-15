from __future__ import annotations

from types import ModuleType
from typing import TypeVar

from .builders import EntityBuilder
from .exceptions import ResourceNotFoundError
from .registry import Registry
from .systems import System, SystemPipeline, SystemState

TResource = TypeVar("TResource")


class World:
    def __init__(self, name: str, global_resources: dict[str, object] | None = None) -> None:
        self.name = name
        self._registry = Registry(global_resources=global_resources)
        self._startup_callbacks: list[object] = []
        self._update_callbacks: list[object] = []
        self._render_callbacks: list[object] = []
        self._shutdown_callbacks: list[object] = []

    def create_entity(self) -> EntityBuilder:
        entity = self._registry.create_entity()
        return EntityBuilder(entity, self._registry)

    def add_system(self, pipeline: SystemPipeline, system: System) -> None:
        self._registry.add_system(pipeline, SystemState.ON, system)

    def add_systems(self, pipeline: SystemPipeline, systems: set[System]) -> None:
        for system in systems:
            self.add_system(pipeline, system)

    def add_system_with_state(
        self,
        pipeline: SystemPipeline,
        system: System,
        state: SystemState,
    ) -> None:
        self._registry.add_system(pipeline, state, system)

    def set_system_state(
        self, pipeline: SystemPipeline, system: System, state: SystemState
    ) -> None:
        self._registry.set_system_state(pipeline, system, state)

    def add_resource(self, resource: object) -> None:
        key = resource.__name__ if isinstance(resource, ModuleType) else type(resource).__name__
        self._registry.resources[key] = resource

    def get_resource(self, resource_type: type[TResource] | ModuleType) -> TResource | ModuleType:
        world_resource = self.get_world_resource(resource_type)
        if world_resource is not None:
            return world_resource
        global_resource = self.get_global_resource(resource_type)
        if global_resource is not None:
            return global_resource
        raise ResourceNotFoundError(getattr(resource_type, "__name__", repr(resource_type)))

    def get_world_resource(
        self,
        resource_type: type[TResource] | ModuleType,
    ) -> TResource | ModuleType | None:
        return _match_resource(self._registry.resources.values(), resource_type)

    def get_global_resource(
        self,
        resource_type: type[TResource] | ModuleType,
    ) -> TResource | ModuleType | None:
        return _match_resource(self._registry.global_resources.values(), resource_type)

    def on_startup(self, callback: object) -> object:
        self._startup_callbacks.append(callback)
        return callback

    def on_update(self, callback: object) -> object:
        self._update_callbacks.append(callback)
        return callback

    def on_render(self, callback: object) -> object:
        self._render_callbacks.append(callback)
        return callback

    def on_shutdown(self, callback: object) -> object:
        self._shutdown_callbacks.append(callback)
        return callback

    def get_registry(self) -> Registry:
        return self._registry


def _match_resource(
    resources: object, resource_type: type[TResource] | ModuleType
) -> TResource | ModuleType | None:
    for resource in resources:
        if isinstance(resource_type, ModuleType):
            if resource is resource_type:
                return resource
            continue
        if isinstance(resource, resource_type):
            return resource
    return None
