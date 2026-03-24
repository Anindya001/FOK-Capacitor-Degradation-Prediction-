"""Minimal placeholder for the deprecated legacy pipeline.

The historical project exposed a large ``legacy_core`` module.  The current
code only requires a callable with a ``run_enhanced_forecast_pipeline`` method
so that the modern pipeline can fall back gracefully when legacy execution
fails.  This stub satisfies the import without pulling in the older code base.
"""

from __future__ import annotations

from typing import Any, Dict


class PICPCore:
    """Placeholder implementation that always raises ``NotImplementedError``."""

    def run_enhanced_forecast_pipeline(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("Legacy pipeline is not available in this build.")


__all__ = ["PICPCore"]
