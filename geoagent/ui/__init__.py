"""UI helpers and exports for the Solara app."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import List, Optional


# Path to the Solara pages directory
PAGES_DIR = str(Path(__file__).parent / "pages")


def launch_ui(extra_args: Optional[List[str]] = None) -> int:
    """Launch the Solara UI for GeoAgent.

    Args:
        extra_args: Additional args passed to `solara run`.

    Returns:
        Process return code.
    """
    cmd = [sys.executable, "-m", "solara", "run", PAGES_DIR]
    if extra_args:
        cmd.extend(extra_args)
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        raise RuntimeError(
            "Solara is not installed. Install with `pip install solara`."
        )


__all__ = ["PAGES_DIR", "launch_ui"]
