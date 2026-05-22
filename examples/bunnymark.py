import random

import numpy as np
from arepy import ArepyEngine, Color, Rect

from arepy_ecs import Component, Entity, Query, VectorValue, With, World
from arepy_ecs.raylib_batch import draw_texture_batch_xy
from arepy_ecs.systems import SystemPipeline


class Vec2(VectorValue):
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


WHITE_COLOR = Color(255, 255, 255, 255)
BUNNY_ASSET = "bunny.png"

BUNNY_COUNT = 10_000
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
SPRITE_SIZE = 32
SPRITE_ORIGIN = 16.0
SPRITE_SOURCE_RECT = Rect(0, 0, SPRITE_SIZE, SPRITE_SIZE)
SPRITE_DRAW_ORIGIN = (SPRITE_ORIGIN, SPRITE_ORIGIN)


class Transform(Component):
    position: Vec2 = Vec2()


class Rigidbody(Component):
    velocity: Vec2 = Vec2()


def movement_system(
    query: Query[Entity, With[Transform, Rigidbody]],
    game: ArepyEngine,
) -> None:
    delta_time = game.renderer_2d.get_delta_time()
    position = query.get_batch(Transform).position
    velocity = query.get_batch(Rigidbody).velocity

    position_x = position.x
    position_y = position.y
    velocity_x = velocity.x
    velocity_y = velocity.y

    position_x += velocity_x * delta_time
    position_y += velocity_y * delta_time

    left = position_x <= 0
    right = position_x >= WINDOW_WIDTH - SPRITE_SIZE
    top = position_y <= 0
    bottom = position_y >= WINDOW_HEIGHT - SPRITE_SIZE

    position_x[left] = 0
    velocity_x[left] = np.abs(velocity_x[left])

    position_x[right] = WINDOW_WIDTH - SPRITE_SIZE
    velocity_x[right] = -np.abs(velocity_x[right])

    position_y[top] = 0
    velocity_y[top] = np.abs(velocity_y[top])

    position_y[bottom] = WINDOW_HEIGHT - SPRITE_SIZE
    velocity_y[bottom] = -np.abs(velocity_y[bottom])


def render_system(
    query: Query[Entity, With[Transform]],
    game: ArepyEngine,
) -> None:
    renderer = game.renderer_2d
    renderer.start_frame()
    renderer.clear(color=WHITE_COLOR)
    texture = game.get_asset_store().get_texture(BUNNY_ASSET)
    position = query.get_batch(Transform).position
    position_x = position.x
    position_y = position.y
    number_of_entities = int(position_x.shape[0])
    draw_texture_batch_xy(
        texture,
        SPRITE_SOURCE_RECT,
        position_x,
        position_y,
        dest_size=(SPRITE_SIZE, SPRITE_SIZE),
        origin=SPRITE_DRAW_ORIGIN,
        rotation=0.0,
        tint=WHITE_COLOR,
    )
    renderer.draw_text(
        f"Entities: {number_of_entities}",
        (10, 30),
        font_size=20,
        color=Color(0, 0, 0, 255),
    )
    renderer.draw_fps((10, 10))
    renderer.end_frame()


def spawn_bunnies(world: World, count: int) -> None:
    for _ in range(count):
        x: float = random.uniform(0, WINDOW_WIDTH - SPRITE_SIZE)
        y: float = random.uniform(0, WINDOW_HEIGHT - SPRITE_SIZE)
        vx: float = random.uniform(-200, 200)
        vy: float = random.uniform(-200, 200)
        world.create_entity().with_component(Transform(position=Vec2(x, y))).with_component(
            Rigidbody(velocity=Vec2(vx, vy))
        ).build()


def main() -> None:
    game: ArepyEngine = ArepyEngine(
        title="Arepy BunnyMark",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        max_frame_rate=10000,
        fullscreen=True,
    )
    asset_store = game.get_asset_store()
    renderer = game.renderer_2d
    asset_store.load_texture(renderer, BUNNY_ASSET, f"./assets/{BUNNY_ASSET}")
    world = World("bunnymark", global_resources={type(game).__name__: game})
    spawn_bunnies(world, BUNNY_COUNT)
    world.add_system(SystemPipeline.UPDATE, movement_system)
    world.add_system(SystemPipeline.RENDER, render_system)

    registry = world.get_registry()
    while not game.display.window_should_close():
        registry.update()
        registry.run(SystemPipeline.UPDATE)
        registry.run(SystemPipeline.RENDER)
        renderer.swap_buffers()


if __name__ == "__main__":
    main()
