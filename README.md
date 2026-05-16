[![CI](https://github.com/scr44gr/arepy-ecs/actions/workflows/ci.yml/badge.svg)](https://github.com/scr44gr/arepy-ecs/actions/workflows/ci.yml)
[![Upload Python Package](https://github.com/scr44gr/arepy-ecs/actions/workflows/python-publish.yml/badge.svg)](https://github.com/scr44gr/arepy-ecs/actions/workflows/python-publish.yml)
[![codecov](https://codecov.io/gh/scr44gr/arepy-ecs/branch/main/graph/badge.svg)](https://codecov.io/gh/scr44gr/arepy-ecs)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# arepy-ecs

`arepy-ecs` is a Rust-backed Entity Component System for arepy.

It is focused on the ECS layer only: worlds, registries, entities, typed components, queries, systems, and zero-copy access to dense component fields through NumPy and Python buffer consumers.

---

## Features

- Public API centered on `World`, `Registry`, `Entity`, `Component`, `System`, `Query`, `With`, and `Without`
- Dense native storage for scalar component fields: `bool`, `int32`, `float32`
- Typed Python component classes with field annotations
- Proxy-based component mutation when accessed through entities and queries
- Zero-copy `memoryview` and NumPy access to component fields
- Batch-oriented systems through `Query.result(...)`
- `uv` + `maturin` + `PyO3` workflow for local development and packaging

---

## Installation

`arepy-ecs` is not published on PyPI yet.

### From GitHub

```bash
pip install git+https://github.com/scr44gr/arepy-ecs.git
```

This builds the native extension locally, so you need a working Rust toolchain.

### Local setup with `uv`

```bash
git clone https://github.com/scr44gr/arepy-ecs.git
cd arepy-ecs
uv sync --dev
uv run maturin develop
```

---

## Quick Start

This example creates a small world, updates a moving entity, and reads the result back through the ECS API.

```python
from dataclasses import dataclass

from arepy_ecs import Component, Entity, Query, With, World
from arepy_ecs.systems import SystemPipeline


class Position(Component):
	x: float = 0.0
	y: float = 0.0


class Velocity(Component):
	x: float = 0.0
	y: float = 0.0


@dataclass(slots=True)
class DeltaTime:
	value: float


def movement_system(query: Query[Entity, With[Position, Velocity]], dt: DeltaTime) -> None:
	for position, velocity in query.iter_components(Position, Velocity):
		position.x += velocity.x * dt.value
		position.y += velocity.y * dt.value


world = World("quickstart")
world.add_resource(DeltaTime(1.0 / 60.0))
world.add_system(SystemPipeline.UPDATE, movement_system)

entity = (world.create_entity()
		  .with_component(Position(x=1.0, y=2.0))
		  .with_component(Velocity(x=0.5, y=-1.0))
		  .build())

registry = world.get_registry()
registry.run(SystemPipeline.UPDATE)

position = entity.get_component(Position)
print(position.x, position.y)
```

---

## Core Concepts

### Entities

Entities are lightweight identifiers created from a `World`:

```python
entity = world.create_entity().build()

player = (world.create_entity()
		  .with_component(Position(x=100.0, y=60.0))
		  .with_component(Velocity(x=2.0, y=0.0))
		  .build())
```

### Components

Components are Python classes declared with type annotations:

```python
from arepy_ecs import Component


class Health(Component):
	current: int = 100
	maximum: int = 100
```

### Systems

Systems are plain functions registered on a pipeline:

```python
def damage_system(query: Query[Entity, With[Health]]) -> None:
	for (health,) in query.iter_components(Health):
		if health.current > 0:
			health.current -= 1


world.add_system(SystemPipeline.UPDATE, damage_system)
```

### Queries

Queries describe component shape with `With[...]` and `Without[...]` filters:

```python
Query[Entity, With[Position, Velocity]]
Query[Entity, Without[Velocity]]
Query[Entity, tuple[With[Position, Velocity], Without[Health]]]
```

Use `iter_components(...)` or `iter_entities_components(...)` for familiar ECS-style iteration.

### Batch Views

When you need dense zero-copy access, `Query.result(...)` returns component batches backed by native storage:

```python
def movement_batch(query: Query[Entity, With[Position, Velocity]], dt: DeltaTime) -> None:
	position, velocity = query.result(Position, Velocity)
	position.x += velocity.x * dt.value
	position.y += velocity.y * dt.value
```

You can also request field-level views directly from the registry:

```python
xs = registry.component_field_ndarray(Position, "x")
raw = registry.component_field_memoryview(Position, "x")
```

Those views are writable and zero-copy. Structural mutation of the underlying component table is blocked while an exported view is still alive, which keeps the Rust storage safe.

### Resources

World resources are injected by type into systems:

```python
@dataclass(slots=True)
class SimulationConfig:
	speed: float


def scale_system(query: Query[Entity, With[Velocity]], config: SimulationConfig) -> None:
	for (velocity,) in query.iter_components(Velocity):
		velocity.x *= config.speed
		velocity.y *= config.speed
```

---

## Testing

```bash
uv run pytest -q
```

To generate the same coverage report used in CI:

```bash
uv run pytest --cov=arepy_ecs --cov-report=xml --cov-report=term
```

For linting and native validation:

```bash
uv run ruff check .
cargo test --workspace
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings -W clippy::pedantic -W clippy::perf
```

---

## Requirements

- Python 3.11+
- Rust stable toolchain
- NumPy 2.1+

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

---

## Acknowledgments

- [Arepy](https://github.com/scr44gr/arepy)
- [PyO3](https://pyo3.rs/)
- [NumPy](https://numpy.org/)
