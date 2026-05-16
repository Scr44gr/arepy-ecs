from __future__ import annotations

from typing import Any, ClassVar, cast, get_type_hints

import numpy as np

Bool = np.bool_
Int32 = np.int32
Float32 = np.float32

_MISSING = object()
_FIELD_KIND_BY_TYPE: dict[Any, str] = {
    bool: "bool",
    int: "int32",
    float: "float32",
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
    "np.bool_": "bool",
    "np.int32": "int32",
    "np.float32": "float32",
    "numpy.bool_": "bool",
    "numpy.int32": "int32",
    "numpy.float32": "float32",
}


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


class Component:
    __ecs_fields__: ClassVar[tuple[str, ...]] = ()
    __ecs_schema__: ClassVar[tuple[tuple[str, str], ...]] = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        hints = {
            name: annotation
            for name, annotation in get_type_hints(cls, include_extras=True).items()
            if not name.startswith("_")
        }
        cls.__ecs_fields__ = tuple(hints)
        cls.__ecs_schema__ = tuple(
            (field_name, _resolve_field_kind(annotation))
            for field_name, annotation in hints.items()
        )

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "_ecs_proxy_enabled", False)
        field_names = type(self).__ecs_fields__
        uses_base_init = type(self).__dict__.get("__init__") is Component.__init__
        if not uses_base_init and not kwargs:
            return

        unexpected = set(kwargs).difference(field_names)
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
            object.__setattr__(self, field_name, value)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        field_names = object.__getattribute__(type(self), "__ecs_fields__")
        proxy_enabled = object.__getattribute__(self, "_ecs_proxy_enabled")
        if proxy_enabled and name in field_names:
            registry = object.__getattribute__(self, "_ecs_proxy_registry")
            entity_id = object.__getattribute__(self, "_ecs_proxy_entity_id")
            return registry._get_component_field(entity_id, type(self), name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        field_names = getattr(type(self), "__ecs_fields__", ())
        proxy_enabled = getattr(self, "_ecs_proxy_enabled", False)
        if proxy_enabled and name in field_names:
            registry = object.__getattribute__(self, "_ecs_proxy_registry")
            entity_id = object.__getattribute__(self, "_ecs_proxy_entity_id")
            registry._set_component_field(entity_id, type(self), name, value)
            return
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name in type(self).__ecs_fields__)
        return f"{type(self).__name__}({fields})"

    def get_id(self) -> int:
        return id(type(self))

    def to_dict(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in type(self).__ecs_fields__}

    @classmethod
    def field_kind(cls, field_name: str) -> str:
        for current_name, kind in cls.__ecs_schema__:
            if current_name == field_name:
                return kind
        raise KeyError(field_name)

    @classmethod
    def make_proxy(cls, registry: Any, entity_id: int) -> Component:
        instance = cls.__new__(cls)
        object.__setattr__(instance, "_ecs_proxy_enabled", True)
        object.__setattr__(instance, "_ecs_proxy_registry", registry)
        object.__setattr__(instance, "_ecs_proxy_entity_id", entity_id)
        return instance

    @classmethod
    def make_batch(cls, registry: Any) -> Component:
        instance = cls.__new__(cls)
        object.__setattr__(instance, "_ecs_proxy_enabled", False)
        for field_name in cls.__ecs_fields__:
            object.__setattr__(
                instance,
                field_name,
                registry.component_field_ndarray(cls, field_name),
            )
        return cast(Component, instance)
