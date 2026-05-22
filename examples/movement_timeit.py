from __future__ import annotations

from dataclasses import dataclass
from timeit import timeit

from arepy_ecs import Component, Entity, Query, VectorValue, With, World
from arepy_ecs.systems import SystemPipeline


class Vec2(VectorValue):
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


class Transform(Component):
    position: Vec2 = Vec2()


class Rigidbody(Component):
    velocity: Vec2 = Vec2()


@dataclass(slots=True)
class DeltaTime:
    value: float


def movement_loop(query: Query[Entity, With[Transform, Rigidbody]], dt: DeltaTime) -> None:
    for transform, rigidbody in query.iter_components(Transform, Rigidbody):
        position = transform.position
        velocity = rigidbody.velocity
        position.x += velocity.x * dt.value
        position.y += velocity.y * dt.value


def movement_batch(query: Query[Entity, With[Transform, Rigidbody]], dt: DeltaTime) -> None:
    transform = query.get_batch(Transform)
    rigidbody = query.get_batch(Rigidbody)
    transform.position.x += rigidbody.velocity.x * dt.value
    transform.position.y += rigidbody.velocity.y * dt.value


def build_world(entity_count: int, system) -> World:
    world = World(system.__name__)
    world.add_resource(DeltaTime(1.0 / 120.0))
    world.add_system(SystemPipeline.UPDATE, system)

    for index in range(entity_count):
        base = float(index)
        world.create_entity().with_component(
            Transform(position=Vec2(base, base * 0.5))
        ).with_component(Rigidbody(velocity=Vec2(1.0, -0.5))).build()

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
    print("movement_batch usa Query.get_batch() sobre el estado actual del World.")
    print("Internamente opera con vistas densas zero-copy para Transform y Rigidbody.")


if __name__ == "__main__":
    main()
