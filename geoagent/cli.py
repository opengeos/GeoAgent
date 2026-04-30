"""GeoAgent CLI.

Commands:
  geoagent ui              Launch the Solara UI
  geoagent codex login     Login with ChatGPT/Codex OAuth
  geoagent --help          Show help
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _run_solara_app() -> int:
    """Launch the Solara UI application."""
    try:
        from geoagent.ui import PAGES_DIR
    except Exception as e:
        print(f"Failed to locate UI app: {e}")
        return 1

    cmd = [sys.executable, "-m", "solara", "run", PAGES_DIR]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("Solara is not installed. Install with `pip install solara`.\n")
        return 1


def _run_codex_login(args: argparse.Namespace) -> int:
    """Run the ChatGPT/Codex OAuth login flow."""
    try:
        from geoagent.core.openai_codex import default_token_file, login_openai_codex

        login_openai_codex(
            open_browser=not args.no_browser,
            timeout_seconds=args.timeout,
            token_file=args.token_file,
        )
        print(f"Saved Codex login to {args.token_file or default_token_file()}")
        return 0
    except Exception as exc:
        print(f"Codex login failed: {exc}")
        return 1


def _run_codex_status(args: argparse.Namespace) -> int:
    """Print local ChatGPT/Codex OAuth status."""
    try:
        from geoagent.core.openai_codex import (
            default_token_file,
            load_token_payload,
            token_expires_soon,
        )

        path = args.token_file or default_token_file()
        payload = load_token_payload(token_file=path)
    except Exception as exc:
        print(f"Not logged in: {exc}")
        return 1

    expires_at = payload.get("expires_at")
    status = "expires soon" if token_expires_soon(expires_at) else "valid"
    if expires_at:
        expiry = time.strftime(
            "%Y-%m-%d %H:%M:%S %Z",
            time.localtime(float(expires_at)),
        )
        print(f"Logged in ({status}); token expires at {expiry}")
    else:
        print(f"Logged in ({status}); token expiry is unknown")
    return 0


def _run_codex_logout(args: argparse.Namespace) -> int:
    """Remove the local ChatGPT/Codex OAuth token file."""
    from geoagent.core.openai_codex import clear_token_payload

    clear_token_payload(token_file=args.token_file)
    print("Removed local Codex login.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the script entry point."""
    parser = argparse.ArgumentParser(
        prog="geoagent",
        description="GeoAgent command line interface",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    subparsers.add_parser("ui", help="Launch the Solara UI")
    codex_parser = subparsers.add_parser(
        "codex",
        help="Manage ChatGPT/Codex OAuth login",
    )
    codex_subparsers = codex_parser.add_subparsers(
        dest="codex_command",
        metavar="command",
    )

    codex_login = codex_subparsers.add_parser(
        "login",
        help="Login with ChatGPT/Codex OAuth",
    )
    codex_login.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the login URL instead of opening a browser",
    )
    codex_login.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Seconds to wait for the browser callback",
    )
    codex_login.add_argument(
        "--token-file",
        help="Override the local token file path",
    )
    codex_login.set_defaults(func=_run_codex_login)

    codex_status = codex_subparsers.add_parser(
        "status",
        help="Show local ChatGPT/Codex OAuth status",
    )
    codex_status.add_argument(
        "--token-file",
        help="Override the local token file path",
    )
    codex_status.set_defaults(func=_run_codex_status)

    codex_logout = codex_subparsers.add_parser(
        "logout",
        help="Remove the local ChatGPT/Codex OAuth token",
    )
    codex_logout.add_argument(
        "--token-file",
        help="Override the local token file path",
    )
    codex_logout.set_defaults(func=_run_codex_logout)

    args = parser.parse_args(argv)

    if args.command == "ui":
        return _run_solara_app()
    if args.command == "codex":
        if hasattr(args, "func"):
            return args.func(args)
        codex_parser.print_help()
        return 0

    # No command: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
