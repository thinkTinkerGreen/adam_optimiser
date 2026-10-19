"""Tests for hyperparameter sweep edge-extension heuristics."""

from __future__ import annotations

from adamlab.sweep import check_and_extend_grid


def test_edge_rule_interior_minimum_no_extension() -> None:
    """When argmin is strictly interior, no extension occurs."""
    grid = [0.001, 0.002, 0.004, 0.008, 0.016]
    losses = [2.5, 2.1, 1.8, 2.0, 2.7]  # argmin at index 2 (0.004)

    extended, new_grid, direction = check_and_extend_grid(grid, losses)
    assert not extended
    assert new_grid == grid
    assert direction == "none"


def test_edge_rule_lower_edge_extension() -> None:
    """When argmin is at index 0, extends 4 points lower."""
    grid = [0.001, 0.002, 0.004]
    losses = [1.5, 2.0, 2.5]  # argmin at 0.001

    extended, new_grid, direction = check_and_extend_grid(grid, losses)
    assert extended
    assert direction == "lower"
    assert len(new_grid) == len(grid) + 4
    assert new_grid[0] < grid[0]


def test_edge_rule_upper_edge_extension() -> None:
    """When argmin is at the last index, extends 4 points higher."""
    grid = [0.001, 0.002, 0.004]
    losses = [2.5, 2.0, 1.5]  # argmin at 0.004

    extended, new_grid, direction = check_and_extend_grid(grid, losses)
    assert extended
    assert direction == "upper"
    assert len(new_grid) == len(grid) + 4
    assert new_grid[-1] > grid[-1]
