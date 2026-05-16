from __future__ import annotations

from typing import TYPE_CHECKING

from .components import Component
from .entities import Entity

if TYPE_CHECKING:
    from .registry import Registry


class EntityBuilder:
    def __init__(self, entity: Entity, registry: Registry) -> None:
        self._entity = entity
        self._registry = registry
        self._components: list[Component] = []

    def with_component(self, component: Component) -> EntityBuilder:
        if not isinstance(component, Component):
            raise TypeError("Component must be of type Component")

        component_type = type(component)
        duplicate_in_builder = any(
            isinstance(current, component_type) for current in self._components
        )
        if duplicate_in_builder or self._entity.has_component(component_type):
            raise TypeError(f"Component `{component_type.__name__}` already exists in entity")

        self._registry.add_component(self._entity, type(component), component)
        self._components.append(component)
        return self

    def build(self) -> Entity:
        return self._entity
