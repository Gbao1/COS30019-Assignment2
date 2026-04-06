from __future__ import annotations

from .common import reconstruct_path
from .models import SearchNode


def format_output(input_name: str, method: str, result: SearchNode | None, nodes_created: int) -> str:
    line1 = f"{input_name} {method.upper()}"
    if result is None:
        line2 = f"NoGoal {nodes_created}"
        line3 = ""
        return "\n".join([line1, line2, line3])

    path = reconstruct_path(result)
    line2 = f"{result.state} {nodes_created}"
    line3 = " ".join(str(node) for node in path)
    return "\n".join([line1, line2, line3])
