from __future__ import annotations

from .components import Component
from .entities import Entity


class EntityBuilder:
    def __init__(self, entity: Entity, registry: object) -> None:
        self._entity = entity
        self._registry = registry

    def with_component(self, component: Component) -> EntityBuilder:
        self._registry.add_component(self._entity, type(component), component)
        return self

    def build(self) -> Entity:
        return self._entity
