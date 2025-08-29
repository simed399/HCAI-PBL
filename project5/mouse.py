from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np

EMPTY = 0
MOUSE = 1
CHEESE = 2
TRAP = 3
WALL = 4
ORG = 5  # organic cheese

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict

class MouseGridEnv:
    """
    5x5 grid. One element per cell: mouse, traps, walls, cheese (normal or organic).
    Rewards:
      +10 on CHEESE or ORG
      -50 on TRAP
      -0.2 on moving to EMPTY or bumping into a WALL (no move)
    """
    def __init__(self, size: int = 5, seed: int | None = None):
        self.size = size
        self.rng = random.Random(seed)
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.mouse_pos: Tuple[int, int] = (0, 0)
        self.max_steps = 60
        self.steps = 0

    def reset(self) -> np.ndarray:
        self._random_layout()
        self.steps = 0
        return self._obs()

    def step(self, action: int) -> StepResult:
        self.steps += 1
        di, dj = ACTIONS[int(action)]
        i, j = self.mouse_pos
        ni, nj = i + di, j + dj

        if not self._in_bounds(ni, nj) or self.grid[ni, nj] == WALL:
            reward = -0.2
            done = False
            prev = WALL
            return StepResult(self._obs(), reward, done, {"prev": int(prev)})

        prev_cell = int(self.grid[ni, nj])
        # move mouse
        self._set_cell(i, j, EMPTY)
        self._set_cell(ni, nj, MOUSE)
        self.mouse_pos = (ni, nj)

        if prev_cell == TRAP:
            reward, done = -50.0, True
        elif prev_cell in (CHEESE, ORG):
            reward, done = 10.0, True
        else:
            reward, done = -0.2, False

        if self.steps >= self.max_steps:
            done = True

        return StepResult(self._obs(), reward, done, {"prev": prev_cell})

    # -------- helpers --------
    def _in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.size and 0 <= j < self.size

    def _set_cell(self, i: int, j: int, val: int) -> None:
        self.grid[i, j] = val

    def _empty_cells(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i, j] == EMPTY]

    def _place_random(self, val: int, k: int) -> None:
        cells = self._empty_cells()
        self.rng.shuffle(cells)
        for r, c in cells[:k]:
            self.grid[r, c] = val

    def _random_layout(self) -> None:
        self.grid.fill(EMPTY)
        self._place_random(WALL, 3)
        self._place_random(TRAP, 2)
        self._place_random(CHEESE, 1)
        self._place_random(ORG, 1)
        # mouse
        r, c = self.rng.choice(self._empty_cells())
        self._set_cell(r, c, MOUSE)
        self.mouse_pos = (r, c)

    def _obs(self) -> np.ndarray:
        obs = np.zeros((6, self.size, self.size), dtype=np.float32)
        for i in range(self.size):
            for j in range(self.size):
                obs[self.grid[i, j], i, j] = 1.0
        return obs
