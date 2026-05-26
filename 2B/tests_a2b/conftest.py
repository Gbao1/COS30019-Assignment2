from __future__ import annotations

import sys
from pathlib import Path


# Allow importing tbrgs when running tests from repository root.
ROOT_2B = Path(__file__).resolve().parent.parent
if str(ROOT_2B) not in sys.path:
    sys.path.insert(0, str(ROOT_2B))
