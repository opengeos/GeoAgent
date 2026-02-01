"""UI helpers and exports for the Streamlit app."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import List, Optional


# Path to the Streamlit app file
APP_PATH = str(Path(__file__).with_name("app.py"))


def launch_ui(extra_args: Optional[List[str]] = None) -> int:
    """Launch the Streamlit UI for GeoAgent.

    Args:
        extra_args: Additional args passed to `streamlit run`.

    Returns:
        Process return code.
    """
    cmd = [sys.executable, "-m", "streamlit", "run", APP_PATH]
    if extra_args:
        cmd.extend(extra_args)
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        raise RuntimeError(
            "Streamlit is not installed. Install with `pip install streamlit`."
        )


__all__ = ["APP_PATH", "launch_ui"]
