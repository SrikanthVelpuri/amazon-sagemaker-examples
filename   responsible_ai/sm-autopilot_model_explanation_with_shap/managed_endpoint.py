"""Deprecated: Use src.autopilot_explainer.endpoint_manager instead."""

import warnings
from src.autopilot_explainer.endpoint_manager import ManagedEndpoint

warnings.warn(
    "managed_endpoint.py is deprecated. Use src.autopilot_explainer.endpoint_manager instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ManagedEndpoint"]
