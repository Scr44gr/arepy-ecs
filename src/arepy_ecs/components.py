from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, cast, get_type_hints

import numpy as np

Bool = np.bool_
Int32 = np.int32
Float32 = np.float32

_MISSING = object()
_FLOAT32_KIND = "float32"
_FIELD_KIND_BY_TYPE: dict[Any, str] = {
    bool: "bool",
    int: "int32",
    float: "float32",
    str: "string",
    Bool: "bool",
    Int32: "int32",
    Float32: "float32",
}
_FIELD_KIND_BY_NAME: dict[str, str] = {
    "bool": "bool",
    "Bool": "bool",
    "int": "int32",
    "Int32": "int32",
    "float": "float32",
    "Float32": "float32",
    "str": "string",
    "String": "string",
    "string": "string",
    "builtins.str": "string",
    "np.bool_": "bool",
    "np.int32": "int32",
    "np.float32": "float32",
    "numpy.bool_": "bool",
    "numpy.int32": "int32",
    "numpy.float32": "float32",
}


def _resolve_vector_axes(vector_type: type[VectorValue]) -> tuple[str, ...]:
    axes = tuple(getattr(vector_type, "__ecs_axes__", ()))
    if axes:
        return axes

    slots = getattr(vector_type, "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    axes = tuple(name for name in slots if not name.startswith("_"))
    if axes:
        return axes

    hints = get_type_hints(vector_type, include_extras=True)
    axes = tuple(name for name in hints if not name.startswith("_"))
    if axes:
        return axes

    raise TypeError(
        f"{vector_type.__name__} must define public vector axes with "
        "__ecs_axes__, __slots__, or type hints"
    )


class VectorValue:
    __slots__ = ()
    __ecs_axes__: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.__ecs_axes__ = _resolve_vector_axes(cls)

    def __iter__(self):
        for axis in type(self).__ecs_axes__:
            yield getattr(self, axis)

    def __repr__(self) -> str:
        values = ", ".join(f"{axis}={getattr(self, axis)!r}" for axis in type(self).__ecs_axes__)
        return f"{type(self).__name__}({values})"

    @classmethod
    def _from_values(cls, values: Sequence[Any]) -> VectorValue:
        if len(values) != len(cls.__ecs_axes__):
            raise TypeError(f"{cls.__name__} expects {len(cls.__ecs_axes__)} values")
        instance = cls.__new__(cls)
        for axis, value in zip(cls.__ecs_axes__, values, strict=False):
            object.__setattr__(instance, axis, value)
        return cast(VectorValue, instance)


_VECTOR_PROXY_TYPES: dict[type[VectorValue], type[VectorValue]] = {}
_VECTOR_ROW_PROXY_TYPES: dict[type[VectorValue], type[VectorValue]] = {}


def _vector_proxy_getattribute(self: VectorValue, name: str) -> Any:
    if name in type(self).__ecs_axes__:
        storage_names = object.__getattribute__(self, "_ecs_proxy_storage_names")
        registry = object.__getattribute__(self, "_ecs_proxy_registry")
        entity_id = object.__getattribute__(self, "_ecs_proxy_entity_id")
        component_type = object.__getattribute__(self, "_ecs_proxy_component_type")
        return registry._get_component_field(
            entity_id, component_type, storage_names[type(self).__ecs_axes__.index(name)]
        )
    return object.__getattribute__(self, name)


def _vector_proxy_setattr(self: VectorValue, name: str, value: Any) -> None:
    if name in type(self).__ecs_axes__:
        storage_names = object.__getattribute__(self, "_ecs_proxy_storage_names")
        registry = object.__getattribute__(self, "_ecs_proxy_registry")
        entity_id = object.__getattribute__(self, "_ecs_proxy_entity_id")
        component_type = object.__getattribute__(self, "_ecs_proxy_component_type")
        registry._set_component_field(
            entity_id, component_type, storage_names[type(self).__ecs_axes__.index(name)], value
        )
        return
    object.__setattr__(self, name, value)


def _vector_row_getattribute(self: VectorValue, name: str) -> Any:
    if name in type(self).__ecs_axes__:
        component = object.__getattribute__(self, "_ecs_row_component")
        field_name = object.__getattribute__(self, "_ecs_field_name")
        batch_component = object.__getattribute__(component, "_ecs_batch_component")
        row_index = object.__getattribute__(component, "_ecs_row_index")
        batch_vector = object.__getattribute__(batch_component, field_name)
        return object.__getattribute__(batch_vector, name)[row_index]
    return object.__getattribute__(self, name)


def _vector_row_setattr(self: VectorValue, name: str, value: Any) -> None:
    if name in type(self).__ecs_axes__:
        component = object.__getattribute__(self, "_ecs_row_component")
        field_name = object.__getattribute__(self, "_ecs_field_name")
        batch_component = object.__getattribute__(component, "_ecs_batch_component")
        row_index = object.__getattribute__(component, "_ecs_row_index")
        batch_vector = object.__getattribute__(batch_component, field_name)
        object.__getattribute__(batch_vector, name)[row_index] = value
        return
    object.__setattr__(self, name, value)


def _get_vector_proxy_type(vector_type: type[VectorValue]) -> type[VectorValue]:
    proxy_type = _VECTOR_PROXY_TYPES.get(vector_type)
    if proxy_type is not None:
        return proxy_type

    proxy_type = cast(
        type[VectorValue],
        type(
            f"_Bound{vector_type.__name__}",
            (vector_type,),
            {
                "__slots__": (
                    "_ecs_proxy_registry",
                    "_ecs_proxy_entity_id",
                    "_ecs_proxy_component_type",
                    "_ecs_proxy_storage_names",
                ),
                "__getattribute__": _vector_proxy_getattribute,
                "__setattr__": _vector_proxy_setattr,
            },
        ),
    )
    _VECTOR_PROXY_TYPES[vector_type] = proxy_type
    return proxy_type


def _get_vector_row_proxy_type(vector_type: type[VectorValue]) -> type[VectorValue]:
    proxy_type = _VECTOR_ROW_PROXY_TYPES.get(vector_type)
    if proxy_type is not None:
        return proxy_type

    proxy_type = cast(
        type[VectorValue],
        type(
            f"_RowBound{vector_type.__name__}",
            (vector_type,),
            {
                "__slots__": ("_ecs_row_component", "_ecs_field_name"),
                "__getattribute__": _vector_row_getattribute,
                "__setattr__": _vector_row_setattr,
            },
        ),
    )
    _VECTOR_ROW_PROXY_TYPES[vector_type] = proxy_type
    return proxy_type


def _resolve_field_kind(annotation: Any) -> str:
    if annotation in _FIELD_KIND_BY_TYPE:
        return _FIELD_KIND_BY_TYPE[annotation]
    if isinstance(annotation, str) and annotation in _FIELD_KIND_BY_NAME:
        return _FIELD_KIND_BY_NAME[annotation]
    module_name = getattr(annotation, "__module__", None)
    type_name = getattr(annotation, "__name__", None)
    if module_name and type_name:
        qualified_name = f"{module_name}.{type_name}"
        if qualified_name in _FIELD_KIND_BY_NAME:
            return _FIELD_KIND_BY_NAME[qualified_name]
    raise TypeError(f"Unsupported component field annotation: {annotation!r}")


def _is_vector_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, VectorValue)


def _coerce_vector_value(vector_type: type[VectorValue], value: Any) -> VectorValue:
    if isinstance(value, vector_type):
        return vector_type._from_values(
            tuple(getattr(value, axis) for axis in vector_type.__ecs_axes__)
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return vector_type._from_values(tuple(value))
    raise TypeError(f"Expected {vector_type.__name__} or a matching sequence, got {value!r}")


def _bind_vector_proxy(
    vector_type: type[VectorValue],
    registry: Any,
    entity_id: int,
    component_type: type[Component],
    storage_names: tuple[str, ...],
) -> VectorValue:
    proxy_type = _get_vector_proxy_type(vector_type)
    instance = proxy_type.__new__(proxy_type)
    object.__setattr__(instance, "_ecs_proxy_registry", registry)
    object.__setattr__(instance, "_ecs_proxy_entity_id", entity_id)
    object.__setattr__(instance, "_ecs_proxy_component_type", component_type)
    object.__setattr__(instance, "_ecs_proxy_storage_names", storage_names)
    return cast(VectorValue, instance)


def _bind_vector_row_proxy(
    vector_type: type[VectorValue],
    component: Component,
    field_name: str,
) -> VectorValue:
    proxy_type = _get_vector_row_proxy_type(vector_type)
    instance = proxy_type.__new__(proxy_type)
    object.__setattr__(instance, "_ecs_row_component", component)
    object.__setattr__(instance, "_ecs_field_name", field_name)
    return cast(VectorValue, instance)


class Component:
    __ecs_fields__: ClassVar[tuple[str, ...]] = ()
    __ecs_schema__: ClassVar[tuple[tuple[str, str], ...]] = ()
    __ecs_field_storage__: ClassVar[dict[str, tuple[str, ...]]] = {}
    __ecs_field_vectors__: ClassVar[dict[str, type[VectorValue]]] = {}
    __ecs_storage_kinds__: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        hints = {
            name: annotation
            for name, annotation in get_type_hints(cls, include_extras=True).items()
            if not name.startswith("_")
        }

        field_storage: dict[str, tuple[str, ...]] = {}
        field_vectors: dict[str, type[VectorValue]] = {}
        storage_kinds: dict[str, str] = {}
        schema: list[tuple[str, str]] = []

        for field_name, annotation in hints.items():
            if _is_vector_type(annotation):
                vector_type = cast(type[VectorValue], annotation)
                storage_names = tuple(f"{field_name}_{axis}" for axis in vector_type.__ecs_axes__)
                field_storage[field_name] = storage_names
                field_vectors[field_name] = vector_type
                for storage_name in storage_names:
                    storage_kinds[storage_name] = _FLOAT32_KIND
                    schema.append((storage_name, _FLOAT32_KIND))
                continue

            kind = _resolve_field_kind(annotation)
            field_storage[field_name] = (field_name,)
            storage_kinds[field_name] = kind
            schema.append((field_name, kind))

        cls.__ecs_fields__ = tuple(hints)
        cls.__ecs_schema__ = tuple(schema)
        cls.__ecs_field_storage__ = field_storage
        cls.__ecs_field_vectors__ = field_vectors
        cls.__ecs_storage_kinds__ = storage_kinds

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "_ecs_proxy_enabled", False)
        object.__setattr__(self, "_ecs_proxy_mode", None)
        object.__setattr__(self, "_ecs_entity_vector_cache", {})
        object.__setattr__(self, "_ecs_row_vector_cache", {})
        field_names = type(self).__ecs_fields__
        uses_base_init = type(self).__dict__.get("__init__") is Component.__init__
        if not uses_base_init and not kwargs:
            return

        unexpected = {
            name for name in kwargs if name not in field_names and not name.startswith("_")
        }
        if unexpected:
            unexpected_names = ", ".join(sorted(unexpected))
            raise TypeError(f"Unexpected component fields: {unexpected_names}")

        for field_name in field_names:
            if field_name in kwargs:
                value = kwargs[field_name]
            else:
                default_value = getattr(type(self), field_name, _MISSING)
                if default_value is _MISSING:
                    raise TypeError(f"Missing component field `{field_name}`")
                value = default_value
            object.__setattr__(self, field_name, self._normalize_public_value(field_name, value))

        for field_name, value in kwargs.items():
            if field_name.startswith("_") and not field_name.startswith("_ecs_"):
                object.__setattr__(self, field_name, value)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_ecs_"):
            return object.__getattribute__(self, name)

        proxy_enabled = object.__getattribute__(self, "_ecs_proxy_enabled")
        proxy_mode = object.__getattribute__(self, "_ecs_proxy_mode") if proxy_enabled else None
        if name.startswith("_"):
            if proxy_enabled:
                registry = object.__getattribute__(self, "_ecs_proxy_registry")
                entity_id = object.__getattribute__(self, "_ecs_proxy_entity_id")
                try:
                    return registry._get_private_component_field(entity_id, type(self), name)
                except AttributeError:
                    pass
            return object.__getattribute__(self, name)

        field_names = object.__getattribute__(type(self), "__ecs_fields__")
        if name not in field_names:
            return object.__getattribute__(self, name)

        vector_types = object.__getattribute__(type(self), "__ecs_field_vectors__")
        vector_type = vector_types.get(name)
        if vector_type is not None:
            if not proxy_enabled:
                return object.__getattribute__(self, name)

            cache_name = "_ecs_row_vector_cache" if proxy_mode == "row" else "_ecs_entity_vector_cache"
            cache = object.__getattribute__(self, cache_name)
            vector_value = cache.get(name)
            if vector_value is None:
                if proxy_mode == "row":
                    vector_value = _bind_vector_row_proxy(
                        vector_type,
                        self,
                        name,
                    )
                else:
                    registry = object.__getattribute__(self, "_ecs_proxy_registry")
                    entity_id = object.__getattribute__(self, "_ecs_proxy_entity_id")
                    storage_names = type(self).__ecs_field_storage__[name]
                    vector_value = _bind_vector_proxy(
                        vector_type,
                        registry,
                        entity_id,
                        type(self),
                        storage_names,
                    )
                cache[name] = vector_value
            return vector_value

        if proxy_enabled:
            registry = object.__getattribute__(self, "_ecs_proxy_registry")
            entity_id = object.__getattribute__(self, "_ecs_proxy_entity_id")
            storage_name = type(self).__ecs_field_storage__[name][0]
            if proxy_mode == "row":
                if type(self).field_kind(storage_name) == "string":
                    return registry._get_component_field(entity_id, type(self), storage_name)
                batch_component = object.__getattribute__(self, "_ecs_batch_component")
                row_index = object.__getattribute__(self, "_ecs_row_index")
                return object.__getattribute__(batch_component, name)[row_index]
            return registry._get_component_field(entity_id, type(self), storage_name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_ecs_"):
            object.__setattr__(self, name, value)
            return

        proxy_enabled = getattr(self, "_ecs_proxy_enabled", False)
        proxy_mode = object.__getattribute__(self, "_ecs_proxy_mode") if proxy_enabled else None
        if name.startswith("_"):
            if proxy_enabled:
                registry = object.__getattribute__(self, "_ecs_proxy_registry")
                entity_id = object.__getattribute__(self, "_ecs_proxy_entity_id")
                registry._set_private_component_field(entity_id, type(self), name, value)
                return
            object.__setattr__(self, name, value)
            return

        field_names = getattr(type(self), "__ecs_fields__", ())
        if name not in field_names:
            object.__setattr__(self, name, value)
            return

        vector_type = getattr(type(self), "__ecs_field_vectors__", {}).get(name)
        if vector_type is not None:
            normalized = _coerce_vector_value(vector_type, value)
            if proxy_enabled:
                if proxy_mode == "row":
                    batch_component = object.__getattribute__(self, "_ecs_batch_component")
                    batch_vector = object.__getattribute__(batch_component, name)
                    row_index = object.__getattribute__(self, "_ecs_row_index")
                    for axis in vector_type.__ecs_axes__:
                        object.__getattribute__(batch_vector, axis)[row_index] = getattr(
                            normalized, axis
                        )
                else:
                    registry = object.__getattribute__(self, "_ecs_proxy_registry")
                    entity_id = object.__getattribute__(self, "_ecs_proxy_entity_id")
                    for axis, storage_name in zip(
                        vector_type.__ecs_axes__, type(self).__ecs_field_storage__[name], strict=False
                    ):
                        registry._set_component_field(
                            entity_id, type(self), storage_name, getattr(normalized, axis)
                        )
                return
            object.__setattr__(self, name, normalized)
            return

        if proxy_enabled:
            registry = object.__getattribute__(self, "_ecs_proxy_registry")
            entity_id = object.__getattribute__(self, "_ecs_proxy_entity_id")
            storage_name = type(self).__ecs_field_storage__[name][0]
            if proxy_mode == "row":
                if type(self).field_kind(storage_name) == "string":
                    registry._set_component_field(entity_id, type(self), storage_name, value)
                else:
                    batch_component = object.__getattribute__(self, "_ecs_batch_component")
                    row_index = object.__getattribute__(self, "_ecs_row_index")
                    object.__getattribute__(batch_component, name)[row_index] = value
                return
            registry._set_component_field(entity_id, type(self), storage_name, value)
            return

        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name in type(self).__ecs_fields__)
        return f"{type(self).__name__}({fields})"

    def get_id(self) -> int:
        return id(type(self))

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for field_name in type(self).__ecs_fields__:
            vector_type = type(self).__ecs_field_vectors__.get(field_name)
            storage_names = type(self).__ecs_field_storage__[field_name]
            if vector_type is None:
                values[storage_names[0]] = getattr(self, field_name)
                continue

            vector_value = _coerce_vector_value(vector_type, getattr(self, field_name))
            for axis, storage_name in zip(vector_type.__ecs_axes__, storage_names, strict=False):
                values[storage_name] = getattr(vector_value, axis)
        return values

    def private_dict(self) -> dict[str, Any]:
        values = getattr(self, "__dict__", {})
        return {
            field_name: value
            for field_name, value in values.items()
            if field_name.startswith("_") and not field_name.startswith("_ecs_")
        }

    @classmethod
    def field_kind(cls, field_name: str) -> str:
        kind = cls.__ecs_storage_kinds__.get(field_name)
        if kind is None:
            raise KeyError(field_name)
        return kind

    @classmethod
    def make_proxy(cls, registry: Any, entity_id: int) -> Component:
        instance = cls.__new__(cls)
        object.__setattr__(instance, "_ecs_proxy_enabled", True)
        object.__setattr__(instance, "_ecs_proxy_mode", "entity")
        object.__setattr__(instance, "_ecs_proxy_registry", registry)
        object.__setattr__(instance, "_ecs_proxy_entity_id", entity_id)
        object.__setattr__(instance, "_ecs_row_index", -1)
        object.__setattr__(instance, "_ecs_batch_component", None)
        object.__setattr__(instance, "_ecs_entity_vector_cache", {})
        object.__setattr__(instance, "_ecs_row_vector_cache", {})
        return cast(Component, instance)

    def bind_row_proxy(self, row_index: int, batch_component: Component) -> None:
        object.__setattr__(self, "_ecs_proxy_mode", "row")
        object.__setattr__(self, "_ecs_row_index", row_index)
        object.__setattr__(self, "_ecs_batch_component", batch_component)

    def bind_entity_proxy(self) -> None:
        object.__setattr__(self, "_ecs_proxy_mode", "entity")
        object.__setattr__(self, "_ecs_row_index", -1)
        object.__setattr__(self, "_ecs_batch_component", None)

    @classmethod
    def make_batch(cls, registry: Any) -> Component:
        return cls._make_batch(registry, registered=False)

    @classmethod
    def _make_batch(cls, registry: Any, *, registered: bool) -> Component:
        instance = cls.__new__(cls)
        object.__setattr__(instance, "_ecs_proxy_enabled", False)
        object.__setattr__(instance, "_ecs_proxy_mode", None)
        object.__setattr__(instance, "_ecs_entity_vector_cache", {})
        object.__setattr__(instance, "_ecs_row_vector_cache", {})
        for field_name in cls.__ecs_fields__:
            object.__setattr__(
                instance,
                field_name,
                cls.make_field_batch(registry, field_name, registered=registered),
            )
        return cast(Component, instance)

    @classmethod
    def make_field_batch(cls, registry: Any, field_name: str, *, registered: bool = False) -> Any:
        storage_names = cls.__ecs_field_storage__.get(field_name)
        if storage_names is None:
            if registered:
                return registry._component_field_ndarray_registered(cls, field_name)
            return registry.component_field_ndarray(cls, field_name)

        vector_type = cls.__ecs_field_vectors__.get(field_name)
        if vector_type is None:
            kind = cls.field_kind(storage_names[0])
            if kind == "string":
                if registered:
                    return registry._component_field_values_registered(cls, storage_names[0])
                return registry.component_field_values(cls, storage_names[0])
            if registered:
                return registry._component_field_ndarray_registered(cls, storage_names[0])
            return registry.component_field_ndarray(cls, storage_names[0])

        return vector_type._from_values(
            tuple(
                registry._component_field_ndarray_registered(cls, storage_name)
                if registered
                else registry.component_field_ndarray(cls, storage_name)
                for storage_name in storage_names
            )
        )

    def _normalize_public_value(self, field_name: str, value: Any) -> Any:
        vector_type = type(self).__ecs_field_vectors__.get(field_name)
        if vector_type is None:
            return value
        return _coerce_vector_value(vector_type, value)
