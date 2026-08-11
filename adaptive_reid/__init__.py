"""Adaptive re-ID protocol infrastructure (STEP 2B).

Implements the frozen 01/01B adaptive re-ID protocol:
    - validation metrics (loss/AUC/accuracy)
    - machine-readable training diagnostics
    - run health classification (NUMERICALLY_INVALID / VALID + near-chance flag)
    - weights-updated detection
    - restart driver (screening 3 / confirmatory 10)
    - staged pipeline (A-E) with no test-derived representative selection
    - final arm summary aggregation (sample SD ddof=1; stub/synthetic metrics blocked)
    - pair-bootstrap policy (R-9 FINAL: patient-cluster estimator withdrawn; pair-level
      bootstrap allowed only as a PAIR-SAMPLING DIAGNOSTIC, not patient-level uncertainty)
    - Top-k frozen gallery/probe infrastructure
    - per-arm determinism check
    - provenance record
"""

from . import constants      # noqa: F401
from . import metrics        # noqa: F401
from . import diagnostics    # noqa: F401
from . import health         # noqa: F401
from . import weights        # noqa: F401
from . import restarts       # noqa: F401
from . import selection      # noqa: F401
from . import summary        # noqa: F401
from . import bootstrap      # noqa: F401
from . import topk           # noqa: F401
from . import determinism    # noqa: F401
from . import provenance     # noqa: F401
from . import pipeline       # noqa: F401

__version__ = '2B.0.0'