"""Procedural outdoor plaza: obstacle map + A* (upgrade path: Nav2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Obstacle:
    name: str
    xy: tuple[float, float]
    radius: float
    kind: str


@dataclass
class WorldData:
    obstacles: list[Obstacle] = field(default_factory=list)
    start_xy: tuple[float, float] = (0.0, 0.0)


def plaza_obstacles() -> list[Obstacle]:
    obs: list[Obstacle] = []
    buildings = [
        ((6.2, 6.0), 1.6, "building"),
        ((-6.0, 6.1), 1.5, "building"),
        ((6.4, -6.0), 1.6, "building"),
        ((-6.3, -5.8), 1.5, "building"),
    ]
    for xy, rad, kind in buildings:
        obs.append(Obstacle(kind, xy, rad, kind))
    for x, y in [(-3.2, 3.4), (3.5, 3.2), (-3.6, -3.3), (3.4, -3.5), (-2.0, 6.5), (2.2, -6.6)]:
        obs.append(Obstacle("tree", (x, y), 0.55, "tree"))
    for x, y in [(-1.8, 2.4), (1.9, -2.3)]:
        obs.append(Obstacle("bench", (x, y), 0.7, "bench"))
    for x, y in [(4.2, 1.2), (-4.3, -1.0), (1.3, 4.4)]:
        obs.append(Obstacle("bin", (x, y), 0.35, "bin"))
    for x, y in [(2.6, 0.9), (-2.7, -0.8)]:
        obs.append(Obstacle("planter", (x, y), 0.40, "planter"))
    return obs


def occupancy_blocked(obstacles: list[Obstacle], x: float, y: float, inflate: float) -> bool:
    for ob in obstacles:
        dx = x - ob.xy[0]
        dy = y - ob.xy[1]
        if dx * dx + dy * dy <= (ob.radius + inflate) ** 2:
            return True
    return False


def astar_path(
    obstacles: list[Obstacle],
    start: tuple[float, float],
    goal: tuple[float, float],
    inflate: float,
    world_size: float = 16.0,
    res: float = 0.25,
) -> list[tuple[float, float]]:
    """Grid A* on a known static map (upgrade path: replace with Nav2)."""
    half = world_size / 2.0 - 0.4
    n = int(world_size / res)

    def w2i(x, y):
        return int((x + half) / res), int((y + half) / res)

    def i2w(i, j):
        return i * res - half, j * res - half

    si, sj = w2i(*start)
    gi, gj = w2i(*goal)
    si, sj = int(np.clip(si, 0, n - 1)), int(np.clip(sj, 0, n - 1))
    gi, gj = int(np.clip(gi, 0, n - 1)), int(np.clip(gj, 0, n - 1))

    blocked = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            x, y = i2w(i, j)
            if occupancy_blocked(obstacles, x, y, inflate):
                blocked[i, j] = True
    blocked[si, sj] = False
    blocked[gi, gj] = False

    import heapq

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    openh = [(h((si, sj), (gi, gj)), 0, (si, sj))]
    came = {}
    gscore = {(si, sj): 0}
    seen = set()
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while openh:
        _, g, cur = heapq.heappop(openh)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == (gi, gj):
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.reverse()
            return [i2w(i, j) for i, j in path]
        for di, dj in dirs:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < n and 0 <= nj < n) or blocked[ni, nj]:
                continue
            ng = g + math.hypot(di, dj)
            if ng < gscore.get((ni, nj), 1e9):
                gscore[(ni, nj)] = ng
                came[(ni, nj)] = cur
                heapq.heappush(openh, (ng + h((ni, nj), (gi, gj)), ng, (ni, nj)))
    return [goal]
