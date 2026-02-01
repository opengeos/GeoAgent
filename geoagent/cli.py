"""GeoAgent CLI.

Commands:
  geoagent ui        Launch the Streamlit UI
  geoagent --help    Show help
"""

from __future__ import annotations

import argparse
import sys
import subprocess


def _run_streamlit_app() -> int:
    try:
        from geoagent.ui import APP_PATH
    except Exception as e:
        print(f"Failed to locate UI app: {e}")
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", APP_PATH]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("Streamlit is not installed. Install with `pip install streamlit`.\n")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="geoagent",
        description="GeoAgent command line interface",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    subparsers.add_parser("ui", help="Launch the Streamlit UI")

    args = parser.parse_args(argv)

    if args.command == "ui":
        return _run_streamlit_app()

    # No command: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

