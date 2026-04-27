"""ppnn: Projection-Pursuit Nearest Neighbors.

A scikit-learn-compatible library implementing the PP-NN family of estimators
based on Touboul's (2010, 2025) phi-divergence projection pursuit.

Three estimators:
  - PPNNHard:  hard-assignment PP-NN (one component label per query)
  - PPNNSoft:  soft, responsibility-weighted PP-NN
  - PPkNN:     hybrid projection pursuit + local k-NN within assigned component

Plus a single dispatch class :class:`PPNNDispatch` that selects one of these
three (or plain k-NN) based on the p-value of the Touboul factorization test.
Author : Touboul Jacques
"""

from .core import TouboulPP, PhiDivergence, DIVERGENCES, cressie_read
from .estimators import (
    PPNNHard,
    PPNNSoft,
    PPkNN,
    PPNNDispatch,
)

__version__ = "0.1.0"

__all__ = [
    "TouboulPP",
    "PhiDivergence",
    "DIVERGENCES",
    "cressie_read",
    "PPNNHard",
    "PPNNSoft",
    "PPkNN",
    "PPNNDispatch",
    "__version__",
]
