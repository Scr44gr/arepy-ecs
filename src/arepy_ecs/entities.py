from __future__ import annotations

from dataclasses import dataclass, field
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
    _component_cache: dict[type[Component], Component] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def get_id(self) -> int:
        return self._entity_id

    def get_component(self, component_type: type[TComponent]) -> TComponent:
        component = self._component_cache.get(component_type)
        if component is None:
            component = self._registry.get_component(self, component_type)
        if component is None:
            raise ComponentNotFoundError(component_type.__name__)
        self._component_cache[component_type] = component
        return component

    def add_component(self, component: Component) -> None:
        self._registry.add_component(self, type(component), component)
        self._component_cache.pop(type(component), None)

    def remove_component(self, component_type: type[TComponent]) -> None:
        self._registry.remove_component(self, component_type)
        self._component_cache.pop(component_type, None)

    def has_component(self, component_type: type[TComponent]) -> bool:
        return self._registry.has_component(self, component_type)

    def kill(self) -> None:
        self._registry.kill_entity(self)
        self._component_cache.clear()

    def __str__(self) -> str:
        return f"Entity(id={self._entity_id})"

    def __repr__(self) -> str:
        return str(self)


class Entities(set):
    """A light runtime-compatible container for Entity instances."""
