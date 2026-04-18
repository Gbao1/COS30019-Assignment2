from __future__ import annotations

from ..common import expand_children
from ..heuristics import heuristic
from ..models import NodeFactory, Problem, SearchNode


def solve_hill_climbing(problem: Problem) -> tuple[SearchNode | None, int]:
    # CUS1: steepest-ascent hill climbing using Euclidean heuristic.
    factory = NodeFactory()
    current = factory.create(problem.origin, None, g_cost=0.0, depth=0)

    while True:
        if current.state in problem.destinations:
            return current, factory.created_count

        current_h = heuristic(problem, current.state)
        best_move: tuple[float, int, float] | None = None

        # Choose the neighbor with the smallest heuristic value.
        # Tie-break by smaller node id, then lower edge cost.
        for child_state, edge_cost in expand_children(problem, current):
            h = heuristic(problem, child_state)
            candidate = (h, child_state, edge_cost)
            if best_move is None or candidate < best_move:
                best_move = candidate

        # No outgoing edge or no strict improvement: local minimum/plateau.
        if best_move is None or best_move[0] >= current_h:
            return None, factory.created_count

        _, next_state, chosen_cost = best_move
        current = factory.create(
            state=next_state,
            parent=current,
            g_cost=current.g_cost + chosen_cost,
            depth=current.depth + 1,
        )


def solve_ids(problem: Problem) -> tuple[SearchNode | None, int]:
    # Backward-compatible alias for existing imports.
    return solve_hill_climbing(problem)
