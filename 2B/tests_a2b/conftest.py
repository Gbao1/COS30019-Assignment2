from __future__ import annotations

import sys
from pathlib import Path


# Ensure tests can import 2B/tbrgs when pytest is run from repository root.
ROOT_2B = Path(__file__).resolve().parent.parent
if str(ROOT_2B) not in sys.path:
    sys.path.insert(0, str(ROOT_2B))
