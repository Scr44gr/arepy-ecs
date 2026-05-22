from __future__ import annotations

from dataclasses import dataclass

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


def movement_system(query: Query[Entity, With[Transform, Rigidbody]], dt: DeltaTime) -> None:
    for transform, rigidbody in query.iter_components(Transform, Rigidbody):
        position = transform.position
        velocity = rigidbody.velocity
        position.x += velocity.x * dt.value
        position.y += velocity.y * dt.value


def build_world(entity_count: int = 5) -> World:
    world = World("movement-system")
    world.add_resource(DeltaTime(1.0 / 60.0))
    world.add_system(SystemPipeline.UPDATE, movement_system)

    for index in range(entity_count):
        world.create_entity().with_component(
            Transform(position=Vec2(float(index), float(index) * 2.0))
        ).with_component(Rigidbody(velocity=Vec2(0.25, -0.5))).build()

    return world


def main() -> None:
    world = build_world()
    registry = world.get_registry()

    registry.run(SystemPipeline.UPDATE)

    query: Query[Entity, With[Transform, Rigidbody]] = Query(include=(Transform, Rigidbody))
    query.set_registry(registry)
    for entity, transform, rigidbody in query.iter_entities_components(Transform, Rigidbody):
        position = transform.position
        velocity = rigidbody.velocity
        print(
            f"entity={entity.get_id()} position=({position.x:.3f}, {position.y:.3f}) "
            f"velocity=({velocity.x:.3f}, {velocity.y:.3f})"
        )


if __name__ == "__main__":
    main()
