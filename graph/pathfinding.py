"""
Advanced Pathfinding Algorithms for Agent Navigation.

Provides optimized implementations of:
- A* with various heuristics
- Dijkstra's algorithm
- BFS/DFS
- Jump Point Search
- Theta* (any-angle pathfinding)
- Flow Field pathfinding
- Memory-efficient path caching
"""

import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import defaultdict, deque
from enum import Enum


class Heuristic(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    OCTILE = "octile"
    COSINE = "cosine"


@dataclass
class PathNode:
    x: int
    y: int
    g: float = float("inf")
    h: float = 0
    f: float = float("inf")
    parent: Optional["PathNode"] = None
    walkable: bool = True

    def __lt__(self, other):
        return self.f < other.f

    @property
    def pos(self):
        return (self.x, self.y)


@dataclass
class PathResult:
    path: list[tuple[int, int]]
    cost: float
    nodes_explored: int
    time_ms: float
    algorithm: str


class PathfindingGrid:
    """Grid-based pathfinding environment."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid: list[list[PathNode]] = []
        for y in range(height):
            row = []
            for x in range(width):
                row.append(PathNode(x=x, y=y))
            self.grid.append(row)

    def set_walkable(self, x: int, y: int, walkable: bool):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].walkable = walkable

    def set_cost(self, x: int, y: int, cost: float):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].g = cost

    def get_node(self, x: int, y: int) -> Optional[PathNode]:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None

    def get_neighbors(self, node: PathNode, allow_diagonal: bool = True) -> list[PathNode]:
        neighbors = []
        # First 4 entries are orthogonal (used when allow_diagonal is False);
        # last 4 entries are diagonal (used for the corner-cutting check via i >= 4).
        dx = [0, -1, 1, 0, -1, 1, -1, 1]
        dy = [-1, 0, 0, 1, -1, -1, 1, 1]

        for i in range(8 if allow_diagonal else 4):
            nx, ny = node.x + dx[i], node.y + dy[i]
            n = self.get_node(nx, ny)
            if n and n.walkable:
                # Check diagonal blocking
                if allow_diagonal and i >= 4:
                    if not self.grid[node.y][nx].walkable or not self.grid[ny][node.x].walkable:
                        continue
                neighbors.append(n)
        return neighbors

    def reset(self):
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x].g = float("inf")
                self.grid[y][x].h = 0
                self.grid[y][x].f = float("inf")
                self.grid[y][x].parent = None


class PathfindingEngine:
    """Optimized pathfinding engine with multiple algorithms."""

    def __init__(self, grid: PathfindingGrid):
        self.grid = grid
        self._cache: dict[tuple, PathResult] = {}
        self._cache_max = 1000

    def find_path_astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        heuristic: Heuristic = Heuristic.EUCLIDEAN,
        allow_diagonal: bool = True,
        max_iterations: int = 10000,
    ) -> Optional[PathResult]:
        """A* pathfinding with configurable heuristic."""
        cache_key = (start, goal, heuristic.value, allow_diagonal)
        if cache_key in self._cache:
            return self._cache[cache_key]

        start_time = time.perf_counter()
        self.grid.reset()

        sx, sy = start
        gx, gy = goal
        start_node = self.grid.get_node(sx, sy)
        goal_node = self.grid.get_node(gx, gy)

        if not start_node or not goal_node or not goal_node.walkable:
            return None

        h_func = self._get_heuristic(heuristic)
        start_node.g = 0
        start_node.h = h_func(start_node, goal_node)
        start_node.f = start_node.h

        open_set = [start_node]
        closed_set = set()
        nodes_explored = 0

        while open_set and nodes_explored < max_iterations:
            current = heapq.heappop(open_set)
            nodes_explored += 1

            if current.pos == goal_node.pos:
                path = self._reconstruct_path(current)
                result = PathResult(
                    path=path,
                    cost=current.g,
                    nodes_explored=nodes_explored,
                    time_ms=(time.perf_counter() - start_time) * 1000,
                    algorithm="A*",
                )
                self._cache_result(cache_key, result)
                return result

            closed_set.add(current.pos)

            for neighbor in self.grid.get_neighbors(current, allow_diagonal):
                if neighbor.pos in closed_set:
                    continue

                # Calculate movement cost
                dx = abs(neighbor.x - current.x)
                dy = abs(neighbor.y - current.y)
                move_cost = math.sqrt(2) if dx + dy == 2 else 1.0
                tentative_g = current.g + move_cost

                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = h_func(neighbor, goal_node)
                    neighbor.f = neighbor.g + neighbor.h
                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)

        return None

    def find_path_dijkstra(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> Optional[PathResult]:
        """Dijkstra's shortest path algorithm."""
        start_time = time.perf_counter()
        self.grid.reset()

        sx, sy = start
        gx, gy = goal
        start_node = self.grid.get_node(sx, sy)
        goal_node = self.grid.get_node(gx, gy)

        if not start_node or not goal_node:
            return None

        start_node.g = 0
        open_set = [(0, start_node)]
        closed_set = set()
        nodes_explored = 0

        while open_set:
            dist, current = heapq.heappop(open_set)
            nodes_explored += 1

            if current.pos == goal_node.pos:
                path = self._reconstruct_path(current)
                return PathResult(
                    path=path,
                    cost=current.g,
                    nodes_explored=nodes_explored,
                    time_ms=(time.perf_counter() - start_time) * 1000,
                    algorithm="Dijkstra",
                )

            if current.pos in closed_set:
                continue
            closed_set.add(current.pos)

            for neighbor in self.grid.get_neighbors(current):
                if neighbor.pos in closed_set:
                    continue
                dx = abs(neighbor.x - current.x)
                dy = abs(neighbor.y - current.y)
                move_cost = math.sqrt(2) if dx + dy == 2 else 1.0
                new_dist = current.g + move_cost
                if new_dist < neighbor.g:
                    neighbor.g = new_dist
                    neighbor.parent = current
                    heapq.heappush(open_set, (new_dist, neighbor))

        return None

    def find_path_bfs(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> Optional[PathResult]:
        """BFS shortest path (unweighted)."""
        start_time = time.perf_counter()
        self.grid.reset()

        sx, sy = start
        gx, gy = goal
        start_node = self.grid.get_node(sx, sy)
        goal_node = self.grid.get_node(gx, gy)

        if not start_node or not goal_node:
            return None

        queue = deque([(start_node, [start_node])])
        visited = {start_node.pos}
        nodes_explored = 0

        while queue:
            current, path = queue.popleft()
            nodes_explored += 1

            if current.pos == goal_node.pos:
                return PathResult(
                    path=[(n.x, n.y) for n in path],
                    cost=len(path) - 1,
                    nodes_explored=nodes_explored,
                    time_ms=(time.perf_counter() - start_time) * 1000,
                    algorithm="BFS",
                )

            for neighbor in self.grid.get_neighbors(current, allow_diagonal=False):
                if neighbor.pos not in visited:
                    visited.add(neighbor.pos)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def find_path_jps(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> Optional[PathResult]:
        """Jump Point Search - optimized A* for uniform grids."""
        start_time = time.perf_counter()
        self.grid.reset()

        sx, sy = start
        gx, gy = goal
        start_node = self.grid.get_node(sx, sy)
        goal_node = self.grid.get_node(gx, gy)

        if not start_node or not goal_node:
            return None

        start_node.g = 0
        start_node.h = math.sqrt((sx - gx) ** 2 + (sy - gy) ** 2)
        start_node.f = start_node.h

        open_set = [start_node]
        closed_set = set()
        nodes_explored = 0

        while open_set:
            current = heapq.heappop(open_set)
            nodes_explored += 1

            if current.pos == goal_node.pos:
                path = self._reconstruct_path(current)
                return PathResult(
                    path=path,
                    cost=current.g,
                    nodes_explored=nodes_explored,
                    time_ms=(time.perf_counter() - start_time) * 1000,
                    algorithm="JPS",
                )

            closed_set.add(current.pos)

            for successor in self._jump_successors(current, goal_node):
                if successor.pos in closed_set:
                    continue
                dx = abs(successor.x - current.x)
                dy = abs(successor.y - current.y)
                move_cost = math.sqrt(dx * dx + dy * dy)
                tentative_g = current.g + move_cost

                if tentative_g < successor.g:
                    successor.parent = current
                    successor.g = tentative_g
                    successor.h = math.sqrt(
                        (successor.x - gx) ** 2 + (successor.y - gy) ** 2
                    )
                    successor.f = successor.g + successor.h
                    if successor not in open_set:
                        heapq.heappush(open_set, successor)

        return None

    def find_path_theta(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> Optional[PathResult]:
        """Theta* - any-angle pathfinding (smoother paths)."""
        start_time = time.perf_counter()
        self.grid.reset()

        sx, sy = start
        gx, gy = goal
        start_node = self.grid.get_node(sx, sy)
        goal_node = self.grid.get_node(gx, gy)

        if not start_node or not goal_node:
            return None

        start_node.g = 0
        start_node.h = math.sqrt((sx - gx) ** 2 + (sy - gy) ** 2)
        start_node.f = start_node.h

        open_set = [start_node]
        closed_set = set()
        nodes_explored = 0

        while open_set:
            current = heapq.heappop(open_set)
            nodes_explored += 1

            if current.pos == goal_node.pos:
                path = self._reconstruct_path(current)
                return PathResult(
                    path=path,
                    cost=current.g,
                    nodes_explored=nodes_explored,
                    time_ms=(time.perf_counter() - start_time) * 1000,
                    algorithm="Theta*",
                )

            closed_set.add(current.pos)

            for neighbor in self.grid.get_neighbors(current):
                if neighbor.pos in closed_set:
                    continue

                # Theta* uses line-of-sight to parent's parent
                if current.parent and self._line_of_sight(
                    current.parent.x, current.parent.y,
                    neighbor.x, neighbor.y
                ):
                    new_g = current.parent.g + math.sqrt(
                        (current.parent.x - neighbor.x) ** 2 +
                        (current.parent.y - neighbor.y) ** 2
                    )
                    if new_g < neighbor.g:
                        neighbor.parent = current.parent
                        neighbor.g = new_g
                        neighbor.h = math.sqrt(
                            (neighbor.x - gx) ** 2 + (neighbor.y - gy) ** 2
                        )
                        neighbor.f = neighbor.g + neighbor.h
                        if neighbor not in open_set:
                            heapq.heappush(open_set, neighbor)
                else:
                    dx = abs(neighbor.x - current.x)
                    dy = abs(neighbor.y - current.y)
                    move_cost = math.sqrt(2) if dx + dy == 2 else 1.0
                    new_g = current.g + move_cost
                    if new_g < neighbor.g:
                        neighbor.parent = current
                        neighbor.g = new_g
                        neighbor.h = math.sqrt(
                            (neighbor.x - gx) ** 2 + (neighbor.y - gy) ** 2
                        )
                        neighbor.f = neighbor.g + neighbor.h
                        if neighbor not in open_set:
                            heapq.heappush(open_set, neighbor)

        return None

    def _jump_successors(self, node: PathNode, goal: PathNode) -> list[PathNode]:
        """Jump to find successor nodes for JPS."""
        successors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dx, dy in directions:
            nx, ny = node.x + dx, node.y + dy
            jump_node = self._jump(nx, ny, dx, dy, goal)
            if jump_node:
                successors.append(jump_node)

        return successors

    def _jump(self, x: int, y: int, dx: int, dy: int, goal: PathNode) -> Optional[PathNode]:
        """Jump in a direction until finding a jump point."""
        node = self.grid.get_node(x, y)
        if not node or not node.walkable:
            return None

        if node.pos == goal.pos:
            return node

        # Diagonal movement
        if dx != 0 and dy != 0:
            # Check for forced neighbors
            if (not self.grid.get_node(x - dx, y) and
                self.grid.get_node(x - dx, y + dy) and
                self.grid.get_node(x - dx, y + dy).walkable):
                return node
            if (not self.grid.get_node(x, y - dy) and
                self.grid.get_node(x + dx, y - dy) and
                self.grid.get_node(x + dx, y - dy).walkable):
                return node

            # Recurse diagonally
            if self._jump(x + dx, y, 0, dy, goal) or self._jump(x, y + dy, dx, 0, goal):
                return node
        else:
            # Horizontal movement
            if dx != 0:
                if (not self.grid.get_node(x, y + 1) and
                    self.grid.get_node(x + dx, y + 1) and
                    self.grid.get_node(x + dx, y + 1).walkable):
                    return node
                if (not self.grid.get_node(x, y - 1) and
                    self.grid.get_node(x + dx, y - 1) and
                    self.grid.get_node(x + dx, y - 1).walkable):
                    return node
            # Vertical movement
            else:
                if (not self.grid.get_node(x + 1, y) and
                    self.grid.get_node(x + 1, y + dy) and
                    self.grid.get_node(x + 1, y + dy).walkable):
                    return node
                if (not self.grid.get_node(x - 1, y) and
                    self.grid.get_node(x - 1, y + dy) and
                    self.grid.get_node(x - 1, y + dy).walkable):
                    return node

        # Continue jumping
        if dx != 0 and dy != 0:
            return self._jump(x + dx, y + dy, dx, dy, goal)
        elif dx != 0:
            return self._jump(x + dx, y, dx, dy, goal)
        else:
            return self._jump(x, y + dy, dx, dy, goal)

    def _line_of_sight(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check line of sight between two points (Bresenham)."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            node = self.grid.get_node(x1, y1)
            if not node or not node.walkable:
                return False
            if x1 == x2 and y1 == y2:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def _get_heuristic(self, heuristic: Heuristic) -> Callable:
        """Get heuristic function."""
        heuristics = {
            Heuristic.EUCLIDEAN: lambda a, b: math.sqrt(
                (a.x - b.x) ** 2 + (a.y - b.y) ** 2
            ),
            Heuristic.MANHATTAN: lambda a, b: abs(a.x - b.x) + abs(a.y - b.y),
            Heuristic.CHEBYSHEV: lambda a, b: max(abs(a.x - b.x), abs(a.y - b.y)),
            Heuristic.OCTILE: lambda a, b: max(abs(a.x - b.x), abs(a.y - b.y)) +
                (math.sqrt(2) - 1) * min(abs(a.x - b.x), abs(a.y - b.y)),
        }
        return heuristics.get(heuristic, heuristics[Heuristic.EUCLIDEAN])

    def _reconstruct_path(self, node: PathNode) -> list[tuple[int, int]]:
        """Reconstruct path from goal to start."""
        path = []
        current = node
        while current:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]

    def _cache_result(self, key: tuple, result: PathResult):
        """Cache path result."""
        if len(self._cache) >= self._cache_max:
            # Remove oldest
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = result

    def clear_cache(self):
        self._cache.clear()

    def compare_algorithms(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> dict[str, PathResult]:
        """Compare all algorithms on the same problem."""
        results = {}
        for algo_name, algo_func in [
            ("A*", self.find_path_astar),
            ("Dijkstra", self.find_path_dijkstra),
            ("BFS", self.find_path_bfs),
            ("JPS", self.find_path_jps),
            ("Theta*", self.find_path_theta),
        ]:
            result = algo_func(start, goal)
            if result:
                results[algo_name] = result
        return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pathfinding algorithms")
    parser.add_argument("--width", type=int, default=50, help="Grid width")
    parser.add_argument("--height", type=int, default=50, help="Grid height")
    parser.add_argument("--start", required=True, help="Start coordinate as x,y")
    parser.add_argument("--goal", required=True, help="Goal coordinate as x,y")
    parser.add_argument(
        "--wall",
        action="append",
        default=[],
        metavar="x,y",
        help="Wall/obstacle coordinate (repeatable)",
    )
    parser.add_argument(
        "--algorithm",
        default="astar",
        choices=["astar", "dijkstra", "bfs", "jps", "theta", "compare"],
        help="Pathfinding algorithm to use",
    )
    parser.add_argument(
        "--heuristic",
        default="euclidean",
        choices=[h.value for h in Heuristic],
        help="Heuristic for A*",
    )
    parser.add_argument(
        "--allow-diagonal",
        dest="allow_diagonal",
        action="store_true",
        default=True,
        help="Allow diagonal movement (default)",
    )
    parser.add_argument(
        "--no-diagonal",
        dest="allow_diagonal",
        action="store_false",
        help="Disable diagonal movement",
    )

    args = parser.parse_args(argv)

    def parse_coord(s):
        x, y = s.split(",")
        return (int(x), int(y))

    try:
        start = parse_coord(args.start)
        goal = parse_coord(args.goal)
        grid = PathfindingGrid(args.width, args.height)
        for w in args.wall:
            wx, wy = parse_coord(w)
            grid.set_walkable(wx, wy, False)
        engine = PathfindingEngine(grid)

        def result_to_dict(r):
            return {
                "path": r.path,
                "cost": r.cost,
                "nodes_explored": r.nodes_explored,
                "time_ms": r.time_ms,
                "algorithm": r.algorithm,
            }

        if args.algorithm == "compare":
            results = engine.compare_algorithms(start, goal)
            data = {name: result_to_dict(r) for name, r in results.items()}
        else:
            algo_map = {
                "astar": engine.find_path_astar,
                "dijkstra": engine.find_path_dijkstra,
                "bfs": engine.find_path_bfs,
                "jps": engine.find_path_jps,
                "theta": engine.find_path_theta,
            }
            func = algo_map[args.algorithm]
            if args.algorithm == "astar":
                result = func(
                    start,
                    goal,
                    heuristic=Heuristic(args.heuristic),
                    allow_diagonal=args.allow_diagonal,
                )
            else:
                result = func(start, goal)

            if result is None:
                print(json.dumps(
                    {"error": "no path found", "start": start, "goal": goal},
                    indent=2,
                ))
                return 1
            data = result_to_dict(result)

        print(json.dumps(data, indent=2, default=str))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
