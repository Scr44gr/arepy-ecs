from __future__ import annotations

import inspect
from dataclasses import dataclass
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    TypeVarTuple,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from ..components import Component
from ..entities import Entity
from .exceptions import QueryDefinitionError

if TYPE_CHECKING:
    from ..registry import Registry

TEntity = TypeVar("TEntity", bound=Entity)
TFilter = TypeVar("TFilter")
Ts = TypeVarTuple("Ts")
TComponent1 = TypeVar("TComponent1", bound=Component)
TComponent2 = TypeVar("TComponent2", bound=Component)
TComponent3 = TypeVar("TComponent3", bound=Component)
TComponent4 = TypeVar("TComponent4", bound=Component)


class With(Generic[*Ts]):
    pass


class Without(Generic[*Ts]):
    pass


@dataclass(frozen=True, slots=True)
class _IterationBindingCache:
    entity_ids: tuple[int, ...]
    entities: tuple[Entity, ...]
    component_rows: tuple[tuple[Component, ...], ...]
    bound_components: tuple[Component, ...]


class Query(Generic[TEntity, TFilter]):
    def __init__(
        self,
        *,
        include: Sequence[type[Any]] = (),
        exclude: Sequence[type[Any]] = (),
    ) -> None:
        self._included = tuple(include)
        self._excluded = tuple(exclude)
        self._registry: Registry | None = None
        self._iteration_cache: dict[tuple[type[Any], ...], _IterationBindingCache] = {}

    def set_registry(self, registry: Registry) -> None:
        if self._registry is not registry:
            self._iteration_cache.clear()
        self._registry = registry

    def get_registry(self) -> Registry:
        if self._registry is None:
            raise RuntimeError("Query is not attached to a Registry")
        return self._registry

    def get_entities(self) -> set[TEntity]:
        return set(self)

    @overload
    def result(self, component_type_1: type[TComponent1], /) -> tuple[TComponent1]: ...

    @overload
    def result(
        self,
        component_type_1: type[TComponent1],
        component_type_2: type[TComponent2],
        /,
    ) -> tuple[TComponent1, TComponent2]: ...

    @overload
    def result(
        self,
        component_type_1: type[TComponent1],
        component_type_2: type[TComponent2],
        component_type_3: type[TComponent3],
        /,
    ) -> tuple[TComponent1, TComponent2, TComponent3]: ...

    @overload
    def result(
        self,
        component_type_1: type[TComponent1],
        component_type_2: type[TComponent2],
        component_type_3: type[TComponent3],
        component_type_4: type[TComponent4],
        /,
    ) -> tuple[TComponent1, TComponent2, TComponent3, TComponent4]: ...

    def result(self, *component_types: type[Component]) -> tuple[Component, ...]:
        selected_types = component_types or cast(tuple[type[Component], ...], self._included)
        registry = self.get_registry()
        return tuple(registry.component_batch(component_type) for component_type in selected_types)

    @overload
    def get_batch(self, component_type: type[TComponent1], /) -> TComponent1: ...

    @overload
    def get_batch(self, component_type: type[Component], field_name: str, /) -> Any: ...

    @overload
    def get_batch(
        self,
        component_type: type[Component],
        field_name: str,
        *field_names: str,
    ) -> tuple[Any, ...]: ...

    def get_batch(self, component_type: type[Component], *field_names: str) -> Any:
        registry = self.get_registry()
        if not field_names:
            return registry.component_batch(component_type)
        if len(field_names) == 1:
            return registry.component_field_batch(component_type, field_names[0])
        return tuple(
            registry.component_field_batch(component_type, field_name) for field_name in field_names
        )

    def add_entity(self, entity: TEntity) -> None:
        _ = entity

    def remove_entity(self, entity: TEntity) -> None:
        _ = entity

    @overload
    def iter_components(self) -> Iterator[tuple[Any, ...]]: ...

    @overload
    def iter_components(
        self,
        component_type_1: type[TComponent1],
        /,
    ) -> Iterator[tuple[TComponent1]]: ...

    @overload
    def iter_components(
        self,
        component_type_1: type[TComponent1],
        component_type_2: type[TComponent2],
        /,
    ) -> Iterator[tuple[TComponent1, TComponent2]]: ...

    @overload
    def iter_components(
        self,
        component_type_1: type[TComponent1],
        component_type_2: type[TComponent2],
        component_type_3: type[TComponent3],
        /,
    ) -> Iterator[tuple[TComponent1, TComponent2, TComponent3]]: ...

    @overload
    def iter_components(
        self,
        component_type_1: type[TComponent1],
        component_type_2: type[TComponent2],
        component_type_3: type[TComponent3],
        component_type_4: type[TComponent4],
        /,
    ) -> Iterator[tuple[TComponent1, TComponent2, TComponent3, TComponent4]]: ...

    def iter_components(self, *component_types: type[Any]) -> Iterator[tuple[Any, ...]]:
        selected_types = component_types or self._included
        for _, components in self._iter_component_rows(selected_types):
            yield components

    @overload
    def iter_entities_components(self) -> Iterator[tuple[TEntity, *tuple[Any, ...]]]: ...

    @overload
    def iter_entities_components(
        self,
        component_type_1: type[TComponent1],
        /,
    ) -> Iterator[tuple[TEntity, TComponent1]]: ...

    @overload
    def iter_entities_components(
        self,
        component_type_1: type[TComponent1],
        component_type_2: type[TComponent2],
        /,
    ) -> Iterator[tuple[TEntity, TComponent1, TComponent2]]: ...

    @overload
    def iter_entities_components(
        self,
        component_type_1: type[TComponent1],
        component_type_2: type[TComponent2],
        component_type_3: type[TComponent3],
        /,
    ) -> Iterator[tuple[TEntity, TComponent1, TComponent2, TComponent3]]: ...

    @overload
    def iter_entities_components(
        self,
        component_type_1: type[TComponent1],
        component_type_2: type[TComponent2],
        component_type_3: type[TComponent3],
        component_type_4: type[TComponent4],
        /,
    ) -> Iterator[tuple[TEntity, TComponent1, TComponent2, TComponent3, TComponent4]]: ...

    def iter_entities_components(self, *component_types: type[Any]) -> Iterator[tuple[Any, ...]]:
        registry = self.get_registry()
        selected_types = component_types or self._included
        for entity_id, components in self._iter_component_rows(selected_types):
            yield (registry._entity(entity_id), *components)

    def _iter_component_rows(
        self,
        selected_types: tuple[type[Any], ...],
    ) -> Iterator[tuple[int, tuple[Any, ...]]]:
        registry = self.get_registry()
        entity_ids = tuple(registry._query_entity_ids(self._included, self._excluded))
        iteration_cache = self._get_iteration_cache(selected_types, entity_ids)
        component_batches = tuple(registry.component_batch(component_type) for component_type in selected_types)

        try:
            for row_index, (entity, components) in enumerate(
                zip(iteration_cache.entities, iteration_cache.component_rows, strict=False)
            ):
                for component, batch_component in zip(components, component_batches, strict=False):
                    component.bind_row_proxy(row_index, batch_component)
                yield (entity.get_id(), components)
        finally:
            for component in iteration_cache.bound_components:
                component.bind_entity_proxy()

    def _get_iteration_cache(
        self,
        selected_types: tuple[type[Any], ...],
        entity_ids: tuple[int, ...],
    ) -> _IterationBindingCache:
        iteration_cache = self._iteration_cache.get(selected_types)
        if iteration_cache is not None and iteration_cache.entity_ids == entity_ids:
            return iteration_cache

        registry = self.get_registry()
        entities = tuple(registry._entity(entity_id) for entity_id in entity_ids)
        component_rows = tuple(
            tuple(entity.get_component(component_type) for component_type in selected_types)
            for entity in entities
        )
        bound_components = tuple(
            component for components in component_rows for component in components
        )
        iteration_cache = _IterationBindingCache(
            entity_ids=entity_ids,
            entities=entities,
            component_rows=component_rows,
            bound_components=bound_components,
        )
        self._iteration_cache[selected_types] = iteration_cache
        return iteration_cache

    def matches(self, entity_signature: Iterable[str]) -> bool:
        current_signature = set(entity_signature)
        included = {component_type.__name__ for component_type in self._included}
        excluded = {component_type.__name__ for component_type in self._excluded}
        return included.issubset(current_signature) and excluded.isdisjoint(current_signature)

    def __iter__(self) -> Iterator[TEntity]:
        if self._registry is None:
            raise RuntimeError("Query is not attached to a Registry")
        yield from cast(
            Iterable[TEntity],
            self._registry.query_entities(self._included, self._excluded),
        )


def _parse_filter_annotation(
    filter_annotation: Any,
) -> tuple[tuple[type[Any], ...], tuple[type[Any], ...]]:
    if filter_annotation in (inspect._empty, Any):
        return (), ()

    items = (
        get_args(filter_annotation)
        if get_origin(filter_annotation) is tuple
        else (filter_annotation,)
    )
    include: list[type[Any]] = []
    exclude: list[type[Any]] = []
    for item in items:
        origin = get_origin(item)
        arguments = get_args(item)
        if origin is With:
            include.extend(arguments)
            continue
        if origin is Without:
            exclude.extend(arguments)
            continue
        raise QueryDefinitionError(f"Unsupported query filter annotation: {item!r}")
    return tuple(include), tuple(exclude)


def build_query_from_annotation(annotation: Any) -> Query[Any, Any]:
    origin = get_origin(annotation)
    if origin is not Query:
        raise QueryDefinitionError(f"Unsupported query annotation: {annotation!r}")
    args = get_args(annotation)
    if len(args) != 2:
        raise QueryDefinitionError(
            f"Query annotation requires two generic arguments: {annotation!r}"
        )
    include, exclude = _parse_filter_annotation(args[1])
    return Query(include=include, exclude=exclude)


def get_signed_query_arguments(function: Any) -> OrderedDict[str, Any]:
    signed_arguments: OrderedDict[str, Any] = OrderedDict()
    resolved_annotations = get_type_hints(function, include_extras=True)
    for parameter in inspect.signature(function).parameters.values():
        annotation = resolved_annotations.get(parameter.name, parameter.annotation)
        if get_origin(annotation) is Query:
            signed_arguments[parameter.name] = annotation
    return signed_arguments


def sign_queries(queries_signature: list[tuple[str, Any]]) -> list[tuple[str, Query[Any, Any]]]:
    return [
        (name, build_query_from_annotation(annotation)) for name, annotation in queries_signature
    ]


def get_queries_instance_from_arguments(args: Sequence[object]) -> list[Query[Any, Any]]:
    return [argument for argument in args if isinstance(argument, Query)]
