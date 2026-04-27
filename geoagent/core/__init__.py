"""GeoAgent core module.

The public surface lives at :mod:`geoagent` and at the per-module
files under :mod:`geoagent.core` (``context``, ``decorators``,
``factory``, ``llm``, ``prompts``, ``registry``, ``result``,
``safety``, ``agent``). This package's ``__init__`` is intentionally
minimal — it does not re-export anything to avoid import-time
side effects.
"""
