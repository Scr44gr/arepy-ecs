from __future__ import annotations

from dataclasses import dataclass
from timeit import timeit

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


def movement_loop(query: Query[Entity, With[Position, Velocity]], dt: DeltaTime) -> None:
    for position, velocity in query.iter_components(Position, Velocity):
        position.x += velocity.x * dt.value
        position.y += velocity.y * dt.value


def movement_batch(query: Query[Entity, With[Position, Velocity]], dt: DeltaTime) -> None:
    position, velocity = query.result(Position, Velocity)
    position.x += velocity.x * dt.value
    position.y += velocity.y * dt.value


def build_world(entity_count: int, system) -> World:
    world = World(system.__name__)
    world.add_resource(DeltaTime(1.0 / 120.0))
    world.add_system(SystemPipeline.UPDATE, system)

    for index in range(entity_count):
        base = float(index)
        world.create_entity().with_component(
            Position(x=base, y=base * 0.5)
        ).with_component(
            Velocity(x=1.0, y=-0.5)
        ).build()

    return world


def benchmark(system, entity_count: int, iterations: int) -> float:
    world = build_world(entity_count, system)
    registry = world.get_registry()
    return timeit(lambda: registry.run(SystemPipeline.UPDATE), number=iterations)


def main() -> None:
    entity_count = 20_000
    iterations = 100

    loop_time = benchmark(movement_loop, entity_count, iterations)
    batch_time = benchmark(movement_batch, entity_count, iterations)

    print(f"entities={entity_count} iterations={iterations}")
    print(f"movement_loop : {loop_time:.6f}s")
    print(f"movement_batch: {batch_time:.6f}s")
    if batch_time > 0:
        print(f"speedup       : {loop_time / batch_time:.2f}x")
    print()
    print("movement_batch usa Query.result() sobre el estado actual del World.")
    print("Internamente opera con vistas densas zero-copy para Position y Velocity.")


if __name__ == "__main__":
    main()