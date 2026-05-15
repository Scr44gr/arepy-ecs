from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from .components import Component
from .exceptions import ComponentNotFoundError

if TYPE_CHECKING:
    from .registry import Registry

TComponent = TypeVar("TComponent", bound=Component)


@dataclass(frozen=True, slots=True)
class Entity:
    _registry: Registry
    _entity_id: int

    def get_id(self) -> int:
        return self._entity_id

    def get_component(self, component_type: type[TComponent]) -> TComponent:
        component = self._registry.get_component(self, component_type)
        if component is None:
            raise ComponentNotFoundError(component_type.__name__)
        return component

    def add_component(self, component: Component) -> None:
        self._registry.add_component(self, type(component), component)

    def remove_component(self, component_type: type[TComponent]) -> None:
        self._registry.remove_component(self, component_type)

    def has_component(self, component_type: type[TComponent]) -> bool:
        return self._registry.has_component(self, component_type)

    def kill(self) -> None:
        self._registry.kill_entity(self)


class Entities(set):
    """A light runtime-compatible container for Entity instances."""
