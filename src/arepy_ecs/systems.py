from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TypeAlias

System: TypeAlias = Callable[..., None]


class SystemPipeline(Enum):
    UPDATE = 0
    RENDER = 1
    INPUT = 2
    PHYSICS = 3
    ASYNC_UPDATE = 4
    RENDER_UI = 5


class SystemState(Enum):
    OFF = 0
    ON = 1
