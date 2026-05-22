from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from collections.abc import Callable
from dataclasses import dataclass
from timeit import timeit

from arepy_ecs import Component, Entity, Query, VectorValue, With, World
from arepy_ecs.systems import SystemPipeline

UpdateSystem = Callable[..., None]


class Vec2(VectorValue):
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


class Position(Component):
    x: float = 0.0
    y: float = 0.0


class Velocity(Component):
    x: float = 0.0
    y: float = 0.0


class Transform(Component):
    position: Vec2 = Vec2()


class Rigidbody(Component):
    velocity: Vec2 = Vec2()


@dataclass(slots=True)
class DeltaTime:
    value: float


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    world_factory: Callable[[int, UpdateSystem], World]
    system: UpdateSystem


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    case_name: str
    elapsed_seconds: float
    tick_milliseconds: float
    nanoseconds_per_entity_update: float
    python_calls_per_tick: int
    python_calls_per_entity: float
    speedup: float
    profile_excerpt: str | None


def scalar_loop(query: Query[Entity, With[Position, Velocity]], dt: DeltaTime) -> None:
    for position, velocity in query.iter_components(Position, Velocity):
        position.x += velocity.x * dt.value
        position.y += velocity.y * dt.value


def scalar_result(query: Query[Entity, With[Position, Velocity]], dt: DeltaTime) -> None:
    position, velocity = query.result(Position, Velocity)
    position.x += velocity.x * dt.value
    position.y += velocity.y * dt.value


def scalar_field_batch(query: Query[Entity, With[Position, Velocity]], dt: DeltaTime) -> None:
    position_x, position_y = query.get_batch(Position, "x", "y")
    velocity_x, velocity_y = query.get_batch(Velocity, "x", "y")
    position_x += velocity_x * dt.value
    position_y += velocity_y * dt.value


def vector_loop(query: Query[Entity, With[Transform, Rigidbody]], dt: DeltaTime) -> None:
    for transform, rigidbody in query.iter_components(Transform, Rigidbody):
        position = transform.position
        velocity = rigidbody.velocity
        position.x += velocity.x * dt.value
        position.y += velocity.y * dt.value


def vector_result(query: Query[Entity, With[Transform, Rigidbody]], dt: DeltaTime) -> None:
    transform, rigidbody = query.result(Transform, Rigidbody)
    transform.position.x += rigidbody.velocity.x * dt.value
    transform.position.y += rigidbody.velocity.y * dt.value


def vector_field_batch(query: Query[Entity, With[Transform, Rigidbody]], dt: DeltaTime) -> None:
    position = query.get_batch(Transform, "position")
    velocity = query.get_batch(Rigidbody, "velocity")
    position.x += velocity.x * dt.value
    position.y += velocity.y * dt.value


def build_scalar_world(entity_count: int, system: UpdateSystem) -> World:
    world = World(system.__name__)
    world.add_resource(DeltaTime(1.0 / 120.0))
    world.add_system(SystemPipeline.UPDATE, system)

    for index in range(entity_count):
        base = float(index)
        world.create_entity().with_component(Position(x=base, y=base * 0.5)).with_component(
            Velocity(x=1.0, y=-0.5)
        ).build()

    return world


def build_vector_world(entity_count: int, system: UpdateSystem) -> World:
    world = World(system.__name__)
    world.add_resource(DeltaTime(1.0 / 120.0))
    world.add_system(SystemPipeline.UPDATE, system)

    for index in range(entity_count):
        base = float(index)
        world.create_entity().with_component(
            Transform(position=Vec2(base, base * 0.5))
        ).with_component(Rigidbody(velocity=Vec2(1.0, -0.5))).build()

    return world


def measure_case(
    case: BenchmarkCase,
    *,
    entity_count: int,
    iterations: int,
    warmup_runs: int,
    profile_top: int,
) -> BenchmarkResult:
    world = case.world_factory(entity_count, case.system)
    registry = world.get_registry()

    for _ in range(warmup_runs):
        registry.run(SystemPipeline.UPDATE)

    profiler = cProfile.Profile()
    profiler.enable()
    registry.run(SystemPipeline.UPDATE)
    profiler.disable()

    profile_stats = pstats.Stats(profiler)
    profile_excerpt: str | None = None
    if profile_top > 0:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(profile_top)
        profile_excerpt = stream.getvalue().rstrip()

    elapsed_seconds = timeit(lambda: registry.run(SystemPipeline.UPDATE), number=iterations)
    tick_milliseconds = elapsed_seconds / iterations * 1_000.0
    nanoseconds_per_entity_update = elapsed_seconds / (iterations * entity_count) * 1_000_000_000.0

    return BenchmarkResult(
        case_name=case.name,
        elapsed_seconds=elapsed_seconds,
        tick_milliseconds=tick_milliseconds,
        nanoseconds_per_entity_update=nanoseconds_per_entity_update,
        python_calls_per_tick=profile_stats.total_calls,
        python_calls_per_entity=profile_stats.total_calls / entity_count,
        speedup=1.0,
        profile_excerpt=profile_excerpt,
    )


def run_section(
    title: str,
    note: str,
    cases: list[BenchmarkCase],
    *,
    entity_count: int,
    iterations: int,
    warmup_runs: int,
    profile_top: int,
) -> list[BenchmarkResult]:
    raw_results = [
        measure_case(
            case,
            entity_count=entity_count,
            iterations=iterations,
            warmup_runs=warmup_runs,
            profile_top=profile_top,
        )
        for case in cases
    ]

    baseline_seconds = raw_results[0].elapsed_seconds
    results = [
        BenchmarkResult(
            case_name=result.case_name,
            elapsed_seconds=result.elapsed_seconds,
            tick_milliseconds=result.tick_milliseconds,
            nanoseconds_per_entity_update=result.nanoseconds_per_entity_update,
            python_calls_per_tick=result.python_calls_per_tick,
            python_calls_per_entity=result.python_calls_per_entity,
            speedup=baseline_seconds / result.elapsed_seconds if result.elapsed_seconds > 0 else 0.0,
            profile_excerpt=result.profile_excerpt,
        )
        for result in raw_results
    ]

    print(title)
    print(note)
    print(
        f"entities={entity_count} iterations={iterations} warmup={warmup_runs} baseline={results[0].case_name}"
    )
    print(
        f"{'case':<18} {'total_s':>12} {'tick_ms':>12} {'ns/update':>14} {'py_calls':>12} {'calls/entity':>14} {'speedup':>10}"
    )
    for result in results:
        print(
            f"{result.case_name:<18} "
            f"{result.elapsed_seconds:>12.6f} "
            f"{result.tick_milliseconds:>12.6f} "
            f"{result.nanoseconds_per_entity_update:>14.2f} "
            f"{result.python_calls_per_tick:>12d} "
            f"{result.python_calls_per_entity:>14.4f} "
            f"{result.speedup:>10.2f}x"
        )
    print()

    if profile_top > 0:
        for result in results:
            print(f"[{title}] profile={result.case_name}")
            print(result.profile_excerpt)
            print()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark movement systems with warmed, apples-to-apples cases.")
    parser.add_argument("--entities", type=int, default=5_000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile-top", type=int, default=0)
    parser.add_argument("--section", choices=("all", "scalar", "vector"), default="all")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()

    scalar_cases = [
        BenchmarkCase("iter_components", build_scalar_world, scalar_loop),
        BenchmarkCase("result", build_scalar_world, scalar_result),
        BenchmarkCase("field_batch", build_scalar_world, scalar_field_batch),
    ]
    vector_cases = [
        BenchmarkCase("iter_components", build_vector_world, vector_loop),
        BenchmarkCase("result", build_vector_world, vector_result),
        BenchmarkCase("field_batch", build_vector_world, vector_field_batch),
    ]

    if arguments.section in ("all", "scalar"):
        run_section(
            "Scalar Components",
            "Fair comparison: same data shape, same update, warmed before timing.",
            scalar_cases,
            entity_count=arguments.entities,
            iterations=arguments.iterations,
            warmup_runs=arguments.warmup,
            profile_top=arguments.profile_top,
        )

    if arguments.section in ("all", "vector"):
        run_section(
            "VectorValue Components",
            "Measures the extra Python/proxy cost of VectorValue on top of the same movement workload.",
            vector_cases,
            entity_count=arguments.entities,
            iterations=arguments.iterations,
            warmup_runs=arguments.warmup,
            profile_top=arguments.profile_top,
        )


if __name__ == "__main__":
    main()