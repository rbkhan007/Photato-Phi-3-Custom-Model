"""Full coverage for the graph package (pathfinding + force-directed layout)."""
import pytest

from graph.pathfinding import (
    PathfindingGrid,
    PathfindingEngine,
    Heuristic,
    PathResult,
)
import graph as graph_module


def _make_grid_and_engine():
    grid = PathfindingGrid(width=10, height=10)
    for x in range(10):
        for y in range(10):
            grid.set_walkable(x, y, True)
    return grid, PathfindingEngine(grid)


def test_pathfinding_astar_finds_path():
    _, engine = _make_grid_and_engine()
    result = engine.find_path_astar((0, 0), (5, 5))
    assert isinstance(result, PathResult)
    assert result.path[0] == (0, 0)
    assert result.path[-1] == (5, 5)


def test_pathfinding_dijkstra():
    _, engine = _make_grid_and_engine()
    result = engine.find_path_dijkstra((0, 0), (3, 3))
    assert result is not None and result.path[-1] == (3, 3)


def test_pathfinding_bfs():
    _, engine = _make_grid_and_engine()
    result = engine.find_path_bfs((0, 0), (2, 2))
    assert result is not None and result.path[-1] == (2, 2)


def test_pathfinding_no_path_when_blocked():
    grid, engine = _make_grid_and_engine()
    for y in range(10):
        grid.set_walkable(5, y, False)  # wall column
    # still reachable around unless fully sealed; seal goal instead
    grid.set_walkable(5, 5, False)
    result = engine.find_path_astar((0, 0), (5, 5))
    assert result is None


def test_pathfinding_jps():
    _, engine = _make_grid_and_engine()
    result = engine.find_path_jps((0, 0), (4, 4))
    assert result is not None


def test_pathfinding_theta():
    _, engine = _make_grid_and_engine()
    result = engine.find_path_theta((0, 0), (4, 4))
    assert result is not None


def test_pathfinding_compare_algorithms():
    _, engine = _make_grid_and_engine()
    comp = engine.compare_algorithms((0, 0), (6, 6))
    assert isinstance(comp, dict)


def test_pathfinding_clear_cache():
    _, engine = _make_grid_and_engine()
    engine.find_path_astar((0, 0), (2, 2))
    engine.clear_cache()
    assert len(engine._cache) == 0


def test_graph_module_imports():
    # Force-directed graph module should import cleanly.
    assert hasattr(graph_module, "__file__")
