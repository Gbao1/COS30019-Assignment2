from __future__ import annotations

from collections import deque
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from route_search.parser import parse_problem


def reachable_goal_exists(problem_file: Path) -> bool:
    problem = parse_problem(problem_file)
    q: deque[int] = deque([problem.origin])
    seen: set[int] = {problem.origin}

    while q:
        cur = q.popleft()
        if cur in problem.destinations:
            return True
        for nxt, _cost in problem.edges.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def edge_lookup(problem_file: Path) -> tuple[dict[int, set[int]], int, set[int]]:
    problem = parse_problem(problem_file)
    edges: dict[int, set[int]] = {}
    for src, outgoing in problem.edges.items():
        edges[src] = {dst for dst, _ in outgoing}
    return edges, problem.origin, set(problem.destinations)


def parse_solver_output(stdout: str) -> tuple[str, int, list[int]]:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Unexpected solver output: {stdout!r}")

    goal_token, created_token = lines[1].split()
    created = int(created_token)
    path: list[int] = []

    if goal_token != "NoGoal":
        if len(lines) < 3:
            raise ValueError(f"Expected path line for solved case: {stdout!r}")
        path = [int(x) for x in lines[2].split()]

    return goal_token, created, path


def main() -> int:
    root = ROOT
    script = root / "search.py"
    cases = sorted((root / "problems" / "test_cases").glob("*.txt"))
    methods = ["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"]

    failures: list[str] = []

    for method in methods:
        for case in cases:
            proc = subprocess.run(
                [sys.executable, str(script), str(case), method],
                cwd=str(root),
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                failures.append(f"{method} {case.name}: non-zero exit\n{proc.stderr.strip()}")
                continue

            goal_token, created, path = parse_solver_output(proc.stdout)
            if created <= 0:
                failures.append(f"{method} {case.name}: created node count not positive ({created})")
                continue

            edges, origin, destinations = edge_lookup(case)
            reachable = reachable_goal_exists(case)

            if goal_token == "NoGoal":
                if reachable:
                    failures.append(f"{method} {case.name}: returned NoGoal but a destination is reachable")
                continue

            goal = int(goal_token)
            if goal not in destinations:
                failures.append(f"{method} {case.name}: returned goal {goal} not in destination set")
                continue

            if not path:
                failures.append(f"{method} {case.name}: empty path for solved case")
                continue

            if path[0] != origin:
                failures.append(f"{method} {case.name}: path does not start at origin ({origin})")
                continue

            if path[-1] != goal:
                failures.append(f"{method} {case.name}: path end ({path[-1]}) does not match goal ({goal})")
                continue

            for a, b in zip(path, path[1:]):
                if b not in edges.get(a, set()):
                    failures.append(f"{method} {case.name}: invalid transition {a}->{b} not in edges")
                    break

    if failures:
        print("VERIFICATION FAILED")
        for item in failures:
            print("-", item)
        return 1

    print(f"VERIFICATION PASSED: {len(methods)} methods x {len(cases)} cases = {len(methods) * len(cases)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
