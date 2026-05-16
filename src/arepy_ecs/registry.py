from __future__ import annotations

import inspect
from types import ModuleType
from typing import Any, TypeVar, get_origin, get_type_hints

import numpy as np

from ._native import FieldView, RawWorld
from .components import Component
from .constants import MAX_COMPONENTS
from .entities import Entity
from .exceptions import MaximumComponentsExceededError, ResourceNotFoundError
from .query import Query, build_query_from_annotation
from .systems import System, SystemPipeline, SystemState

TComponent = TypeVar("TComponent", bound=Component)

_NUMPY_DTYPES = {
    "bool": np.bool_,
    "int32": np.int32,
    "float32": np.float32,
}


class Registry:
    def __init__(self, global_resources: dict[str, object] | None = None) -> None:
        self._native = RawWorld()
        self._entity_cache: dict[int, Entity] = {}
        self._component_types: dict[str, type[Component]] = {}
        self._systems: dict[SystemPipeline, dict[SystemState, dict[System, None]]] = {
            pipeline: {state: {} for state in SystemState} for pipeline in SystemPipeline
        }
        self.resources: dict[str, object] = {}
        self.global_resources: dict[str, object] = dict(global_resources or {})

    @property
    def number_of_entities(self) -> int:
        return self._native.alive_count()

    @property
    def number_of_systems(self) -> int:
        return sum(len(bucket) for states in self._systems.values() for bucket in states.values())

    @property
    def systems(self) -> dict[SystemPipeline, dict[SystemState, set[System]]]:
        return {
            pipeline: {state: set(bucket) for state, bucket in states.items()}
            for pipeline, states in self._systems.items()
        }

    def create_entity(self) -> Entity:
        entity_id = self._native.create_entity()
        return self._entity(entity_id)

    def kill_entity(self, entity: Entity) -> None:
        self._native.kill_entity(entity.get_id())
        self._entity_cache.pop(entity.get_id(), None)

    def add_component(
        self,
        entity: Entity,
        component_type: type[TComponent],
        component: TComponent,
        sync_queries: bool = False,
    ) -> None:
        _ = sync_queries
        self._ensure_component_registered(component_type)
        self._native.add_component(entity.get_id(), component_type.__name__, component.to_dict())

    def get_component(self, entity: Entity, component_type: type[TComponent]) -> TComponent | None:
        self._ensure_component_registered(component_type)
        if not self.has_component(entity, component_type):
            return None
        return component_type.make_proxy(self, entity.get_id())  # type: ignore

    def remove_component(self, entity: Entity, component_type: type[TComponent]) -> None:
        self._ensure_component_registered(component_type)
        self._native.remove_component(entity.get_id(), component_type.__name__)

    def has_component(self, entity: Entity, component_type: type[TComponent]) -> bool:
        self._ensure_component_registered(component_type)
        return self._native.has_component(entity.get_id(), component_type.__name__)

    def add_system(self, pipeline: SystemPipeline, state: SystemState, system: System) -> None:
        for current_state in SystemState:
            self._systems[pipeline][current_state].pop(system, None)
        self._systems[pipeline][state][system] = None

    def set_system_state(
        self, pipeline: SystemPipeline, system: System, state: SystemState
    ) -> None:
        self.add_system(pipeline, state, system)

    def update(self) -> None:
        return None

    def run(self, pipeline: SystemPipeline) -> None:
        for system in list(self._systems[pipeline][SystemState.ON]):
            self._invoke_system(system)

    def get_resource(self, resource_name: str) -> object | None:
        return self.resources.get(resource_name) or self.global_resources.get(resource_name)

    def query_entities(
        self,
        include_types: tuple[type[Any], ...],
        exclude_types: tuple[type[Any], ...],
    ) -> list[Entity]:
        include_names = [component_type.__name__ for component_type in include_types]
        exclude_names = [component_type.__name__ for component_type in exclude_types]
        return [
            self._entity(entity_id)
            for entity_id in self._native.query_entities(include_names, exclude_names)
        ]

    def component_field_ndarray(
        self, component_type: type[Component], field_name: str
    ) -> np.ndarray:
        view = self.component_field_view(component_type, field_name)
        dtype = _NUMPY_DTYPES[component_type.field_kind(field_name)]
        return np.frombuffer(view, dtype=dtype, count=len(view))  # type: ignore

    def component_field_array(self, component_type: type[Component], field_name: str) -> np.ndarray:
        return self.component_field_ndarray(component_type, field_name)

    def component_batch(self, component_type: type[TComponent]) -> TComponent:
        self._ensure_component_registered(component_type)
        return component_type.make_batch(self)  # type: ignore

    def component_field_view(self, component_type: type[Component], field_name: str) -> FieldView:
        self._ensure_component_registered(component_type)
        return self._native.component_field_view(component_type.__name__, field_name)

    def component_field_memoryview(
        self,
        component_type: type[Component],
        field_name: str,
    ) -> memoryview:
        return memoryview(self.component_field_view(component_type, field_name))  # type: ignore

    def _get_component_field(
        self,
        entity_id: int,
        component_type: type[Component],
        field_name: str,
    ) -> Any:
        self._ensure_component_registered(component_type)
        return self._native.get_component_field(entity_id, component_type.__name__, field_name)

    def _set_component_field(
        self,
        entity_id: int,
        component_type: type[Component],
        field_name: str,
        value: Any,
    ) -> None:
        self._ensure_component_registered(component_type)
        self._native.set_component_field(entity_id, component_type.__name__, field_name, value)

    def _entity(self, entity_id: int) -> Entity:
        entity = self._entity_cache.get(entity_id)
        if entity is None:
            entity = Entity(self, entity_id)
            self._entity_cache[entity_id] = entity
        return entity

    def _ensure_component_registered(self, component_type: type[Component]) -> None:
        component_name = component_type.__name__
        if component_name in self._component_types:
            return
        if len(self._component_types) >= MAX_COMPONENTS:
            raise MaximumComponentsExceededError(MAX_COMPONENTS)
        self._native.register_component(
            component_name,
            list(component_type.__ecs_schema__),
        )
        self._component_types[component_name] = component_type

    def _invoke_system(self, system: System) -> None:
        arguments: list[object] = []
        resolved_annotations = get_type_hints(system, include_extras=True)
        for parameter in inspect.signature(system).parameters.values():
            annotation = resolved_annotations.get(parameter.name, parameter.annotation)
            if get_origin(annotation) is Query:
                query = build_query_from_annotation(annotation)
                query.set_registry(self)
                arguments.append(query)
                continue
            arguments.append(self._resolve_resource(parameter.name, annotation))
        system(*arguments)

    def _resolve_resource(self, parameter_name: str, annotation: Any) -> object:
        if annotation is inspect._empty:
            resource = self.resources.get(parameter_name) or self.global_resources.get(
                parameter_name
            )
            if resource is None:
                raise ResourceNotFoundError(parameter_name)
            return resource

        for resource in (*self.resources.values(), *self.global_resources.values()):
            if isinstance(annotation, ModuleType) and resource is annotation:
                return resource
            if isinstance(annotation, type) and isinstance(resource, annotation):
                return resource
            if resource is annotation:
                return resource
        raise ResourceNotFoundError(getattr(annotation, "__name__", parameter_name))
