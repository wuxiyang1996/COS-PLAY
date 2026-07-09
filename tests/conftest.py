"""Top-level pytest config — make project root importable."""

import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__) + os.sep + os.pardir)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
