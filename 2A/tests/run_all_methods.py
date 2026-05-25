from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    script = root / "search.py"
    problem = root / "problems" / "sample_problem.txt"

    methods = ["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"]
    for method in methods:
        print(f"\n--- {method} ---")
        completed = subprocess.run(
            [sys.executable, str(script), str(problem), method],
            cwd=str(root),
            text=True,
            capture_output=True,
            check=False,
        )
        print(completed.stdout.strip())
        if completed.returncode != 0:
            print(completed.stderr.strip())
            return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
