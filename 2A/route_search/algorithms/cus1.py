from __future__ import annotations

from ..common import expand_children
from ..heuristics import heuristic
from ..models import NodeFactory, Problem, SearchNode


def solve_hill_climbing(problem: Problem, max_sideways: int = 100) -> tuple[SearchNode | None, int]:
    # CUS1: steepest-ascent hill climbing using Euclidean heuristic.
    factory = NodeFactory()
    current = factory.create(problem.origin, None, g_cost=0.0, depth=0)
    sideways_moves = 0

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

        if best_move is None:
            return None, factory.created_count

        best_h, next_state, chosen_cost = best_move

        # PLATEAU MITIGATION LOGIC
        if best_h < current_h:
            # Found a strict improvement, reset the sideways counter.
            sideways_moves = 0
        elif best_h == current_h:
            # On a plateau: allow only a bounded number of sideways moves.
            if sideways_moves >= max_sideways:
                return None, factory.created_count
            sideways_moves += 1
        else:
            # Neighbor is worse (local minimum).
            return None, factory.created_count

        current = factory.create(
            state=next_state,
            parent=current,
            g_cost=current.g_cost + chosen_cost,
            depth=current.depth + 1,
        )
