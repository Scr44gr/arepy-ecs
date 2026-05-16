from __future__ import annotations

import inspect
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

    def set_registry(self, registry: Registry) -> None:
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

    def add_entity(self, entity: TEntity) -> None:
        _ = entity

    def remove_entity(self, entity: TEntity) -> None:
        _ = entity

    def iter_components(self, *component_types: type[Any]) -> Iterator[tuple[Any, ...]]:
        selected_types = component_types or self._included
        for entity in self:
            yield tuple(entity.get_component(component_type) for component_type in selected_types)

    def iter_entities_components(self, *component_types: type[Any]) -> Iterator[tuple[Any, ...]]:
        selected_types = component_types or self._included
        for entity in self:
            components = tuple(
                entity.get_component(component_type) for component_type in selected_types
            )
            yield (entity, *components)

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
