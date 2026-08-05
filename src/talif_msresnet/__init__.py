"""TA-LIF reproducibility reference implementation for directly trained SNNs.

This package is a new reference implementation reconstructed from the manuscript
and dissertation descriptions. It is not the original dissertation code.
"""

__version__ = "1.0.0"

__all__ = ["build_model", "__version__"]


def __getattr__(name: str):
    if name == "build_model":
        from .models import build_model

        return build_model
    raise AttributeError(name)
