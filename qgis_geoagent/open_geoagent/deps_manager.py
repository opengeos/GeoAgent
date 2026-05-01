"""
Dependency Manager for OpenGeoAgent

Manages a virtual environment for plugin dependencies
to avoid polluting the QGIS built-in Python environment.

The venv is created at ~/.open_geoagent/venv_pyX.Y and its
site-packages directory is added to sys.path at runtime.

All ``subprocess`` calls in this module use list-form argv built from
internal constants (the resolved Python interpreter, the resolved ``uv``
binary, fixed flags) and never accept user-supplied input or run with
``shell=True``. The ``# nosec`` annotations on the import and call sites
document this explicitly for the plugins.qgis.org Bandit scan.
"""

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess  # nosec B404
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

from qgis.PyQt.QtCore import QThread, pyqtSignal

# Dependency specs: (import_name, pip_install_name)
MIN_PYTHON_VERSION = (3, 11)
CORE_RUNTIME_PACKAGES = [
    ("geoagent", "GeoAgent[providers]>=1.4.1"),
    ("strands", "strands-agents>=1.37"),
    ("pydantic", "pydantic>=2.0"),
]
PROVIDER_PACKAGES = [
    ("openai", "openai>=1.0"),
    ("anthropic", "anthropic>=0.40"),
    ("google.genai", "google-genai>=1.0"),
    ("ollama", "ollama>=0.3"),
    ("litellm", "strands-agents[litellm]>=1.37"),
]
WHITEBOX_PACKAGES = [("whitebox", "whitebox>=2.3.6")]
NASA_PACKAGES = [("earthaccess", "earthaccess>=0.10")]
GEE_PACKAGES = [
    ("ee", "earthengine-api>=1.0"),
    ("geemap", "geemap"),
]
STAC_PACKAGES = [
    ("pystac_client", "pystac-client>=0.8"),
    ("planetary_computer", "planetary-computer>=1.0"),
]

DEPENDENCY_GROUPS = {
    "Core Providers": CORE_RUNTIME_PACKAGES + PROVIDER_PACKAGES,
    "WhiteboxTools": CORE_RUNTIME_PACKAGES + WHITEBOX_PACKAGES,
    "NASA Earthdata/OPERA": CORE_RUNTIME_PACKAGES + NASA_PACKAGES,
    "GEE Data Catalogs": CORE_RUNTIME_PACKAGES + GEE_PACKAGES,
    "STAC": CORE_RUNTIME_PACKAGES + STAC_PACKAGES,
}


def _dedupe_packages(packages: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return dependency specs with duplicate import names removed."""
    seen = set()
    out = []
    for import_name, pip_name in packages:
        if import_name in seen:
            continue
        seen.add(import_name)
        out.append((import_name, pip_name))
    return out


DEPENDENCY_GROUPS["All"] = _dedupe_packages(
    [
        *CORE_RUNTIME_PACKAGES,
        *PROVIDER_PACKAGES,
        *WHITEBOX_PACKAGES,
        *NASA_PACKAGES,
        *GEE_PACKAGES,
        *STAC_PACKAGES,
    ]
)

# Backwards-compatible name used by tests and settings UI. These are the
# packages needed for the default provider surface, not every optional mode.
REQUIRED_PACKAGES = DEPENDENCY_GROUPS["Core Providers"]

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".open_geoagent")
PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"


def get_venv_dir() -> str:
    """Get the path to the plugin's virtual environment directory.

    Returns:
        Path to the venv directory (~/.open_geoagent/venv_pyX.Y).
    """
    return os.path.join(CACHE_DIR, f"venv_{PYTHON_VERSION}")


def get_venv_python_path(venv_dir: Optional[str] = None) -> str:
    """Get the path to the Python executable inside the venv.

    Args:
        venv_dir: Path to the venv directory. Defaults to get_venv_dir().

    Returns:
        Path to the venv's Python executable.
    """
    if venv_dir is None:
        venv_dir = get_venv_dir()
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python3")


def get_venv_site_packages(venv_dir: Optional[str] = None) -> str:
    """Get the path to the venv's site-packages directory.

    Args:
        venv_dir: Path to the venv directory. Defaults to get_venv_dir().

    Returns:
        Path to the venv's site-packages directory.
    """
    if venv_dir is None:
        venv_dir = get_venv_dir()
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Lib", "site-packages")

    # On Unix, detect the actual Python version directory
    lib_dir = os.path.join(venv_dir, "lib")
    if os.path.isdir(lib_dir):
        for entry in sorted(os.listdir(lib_dir)):
            if entry.startswith("python"):
                candidate = os.path.join(lib_dir, entry, "site-packages")
                if os.path.isdir(candidate):
                    return candidate

    # Fallback using current Python version
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return os.path.join(venv_dir, "lib", py_ver, "site-packages")


def venv_exists() -> bool:
    """Check if the plugin's virtual environment exists and has a Python executable.

    Returns:
        True if the venv directory and Python executable exist.
    """
    venv_dir = get_venv_dir()
    python_path = get_venv_python_path(venv_dir)
    return os.path.isdir(venv_dir) and os.path.isfile(python_path)


def ensure_venv_packages_available() -> bool:
    """Add the venv's site-packages to sys.path if the venv exists.

    This is safe to call multiple times (idempotent). If the venv does not
    exist yet, this is a no-op.

    Returns:
        True if site-packages was added or already present, False if venv
        does not exist.
    """
    if not venv_exists():
        return False

    site_packages = get_venv_site_packages()
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
    return True


def _dependency_discoverable(import_name: str) -> Tuple[bool, Optional[str]]:
    """Return whether a dependency is discoverable on ``sys.path``.

    The plugin uses this lightweight check while opening the chat dock. Some
    provider modules are expensive to import, so startup should only confirm
    they are installed and leave full imports to the actual chat request.

    Args:
        import_name: Dotted module name to probe.

    Returns:
        Tuple of (discoverable, error_message). ``error_message`` is ``None``
        when discoverable is ``True``.
    """
    try:
        if importlib.util.find_spec(import_name) is None:
            return False, f"No module named {import_name!r}"
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _distribution_name(pip_name: str) -> str:
    """Return a best-effort distribution name from a pip requirement string."""
    return (
        pip_name.split("[", 1)[0]
        .split(">", 1)[0]
        .split("<", 1)[0]
        .split("=", 1)[0]
        .strip()
    )


def _dependency_version(pip_name: str) -> Optional[str]:
    """Return an installed distribution version without importing the package."""
    try:
        return importlib.metadata.version(_distribution_name(pip_name))
    except importlib.metadata.PackageNotFoundError:
        return None


def python_runtime_supported() -> bool:
    """Return True when the current Python can install/run GeoAgent."""
    return sys.version_info >= MIN_PYTHON_VERSION


def python_runtime_error() -> str:
    """Return a clear unsupported-runtime message for QGIS users."""
    required = ".".join(str(part) for part in MIN_PYTHON_VERSION)
    current = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    return (
        f"OpenGeoAgent requires Python {required} or newer because GeoAgent "
        f"requires Python >=3.11. This QGIS session is using Python {current}. "
        "Install a newer QGIS build, or use a QGIS Python environment based on "
        f"Python {required}+."
    )


def dependency_group_names() -> List[str]:
    """Return dependency group names in UI order."""
    return list(DEPENDENCY_GROUPS.keys())


def packages_for_group(group_name: str = "Core Providers") -> List[Tuple[str, str]]:
    """Return dependency specs for one named group."""
    return list(DEPENDENCY_GROUPS.get(group_name, REQUIRED_PACKAGES))


def check_dependencies(group_name: str = "Core Providers") -> List[Dict]:
    """Check if required Python packages are discoverable.

    This intentionally avoids importing provider modules so opening the
    settings panel stays responsive.

    Returns:
        List of dicts with keys: name, pip_name, installed, version, error.
        ``error`` is populated when the package is not discoverable.
    """
    ensure_venv_packages_available()
    results = []
    for import_name, pip_name in packages_for_group(group_name):
        info: Dict = {
            "name": import_name,
            "pip_name": pip_name,
            "installed": False,
            "version": None,
            "error": None,
        }
        discoverable, error = _dependency_discoverable(import_name)
        if discoverable:
            info["installed"] = True
            info["version"] = _dependency_version(pip_name) or "installed"
        else:
            info["error"] = error
        results.append(info)
    return results


def all_dependencies_met() -> bool:
    """Return True if core runtime packages are discoverable.

    This is used on the chat-dock opening path, so it must avoid importing
    heavy provider packages or optional workflow stacks. Full imports happen
    later when the user sends a chat request, where provider/mode-specific
    import errors can be reported in the chat panel.

    Returns:
        True if core dependencies are discoverable on ``sys.path``.
    """
    ensure_venv_packages_available()
    return all(
        _dependency_discoverable(import_name)[0]
        for import_name, _ in CORE_RUNTIME_PACKAGES
    )


def get_missing_packages(group_name: str = "Core Providers") -> List[str]:
    """Return pip install names of missing packages.

    Returns:
        List of pip package names that are not currently importable.
    """
    return [
        dep["pip_name"]
        for dep in check_dependencies(group_name)
        if not dep["installed"]
    ]


def _get_clean_env() -> dict:
    """Get a clean copy of the environment for subprocess calls.

    Removes variables that could interfere with venv creation and pip installs.

    Returns:
        A copy of os.environ with problematic variables removed.
    """
    env = os.environ.copy()
    for var in [
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "QGIS_PREFIX_PATH",
        "QGIS_PLUGINPATH",
    ]:
        env.pop(var, None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _get_subprocess_kwargs() -> dict:
    """Get platform-specific subprocess keyword arguments.

    On Windows, suppresses the console window that would otherwise pop up
    for each subprocess invocation.

    Returns:
        Dict of kwargs to pass to subprocess.run().
    """
    if platform.system() == "Windows":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _uv_usable() -> bool:
    """Return True when the cached uv binary exists and verifies successfully."""
    try:
        from .uv_manager import uv_exists, verify_uv

        if not uv_exists():
            return False
        success, _message = verify_uv()
        return bool(success)
    except Exception:
        return False


def _find_python_executable() -> str:
    """Find a working Python executable for subprocess calls.

    On QGIS Windows, ``sys.executable`` may point to ``qgis-bin.exe`` rather
    than a Python interpreter, which would launch another QGIS instance when
    used in subprocess calls. This function searches for the actual Python
    executable using multiple strategies.

    Returns:
        Path to a Python executable, or ``sys.executable`` as fallback.
    """
    if platform.system() != "Windows":
        return sys.executable

    # Strategy 1: Check if sys.executable is already Python
    exe_name = os.path.basename(sys.executable).lower()
    if exe_name in ("python.exe", "python3.exe"):
        return sys.executable

    # Strategy 2: Use sys._base_prefix to find the Python installation.
    # On QGIS Windows, sys._base_prefix typically points to
    # C:\Program Files\QGIS 3.x\apps\Python3x\
    base_prefix = getattr(sys, "_base_prefix", None) or sys.prefix
    python_in_prefix = os.path.join(base_prefix, "python.exe")
    if os.path.isfile(python_in_prefix):
        return python_in_prefix

    # Strategy 3: Look for python.exe next to sys.executable
    exe_dir = os.path.dirname(sys.executable)
    for name in ("python.exe", "python3.exe"):
        candidate = os.path.join(exe_dir, name)
        if os.path.isfile(candidate):
            return candidate

    # Strategy 4: Walk up from sys.executable to find apps/Python3x/python.exe
    # Typical QGIS layout: .../QGIS 3.x/bin/qgis-bin.exe
    #                       .../QGIS 3.x/apps/Python3x/python.exe
    parent = os.path.dirname(exe_dir)
    apps_dir = os.path.join(parent, "apps")
    if os.path.isdir(apps_dir):
        best_candidate = None
        best_version_num = -1
        for entry in os.listdir(apps_dir):
            lower_entry = entry.lower()
            if not lower_entry.startswith("python"):
                continue
            suffix = lower_entry.removeprefix("python")
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if not digits:
                continue
            try:
                version_num = int(digits)
            except ValueError:
                continue
            candidate = os.path.join(apps_dir, entry, "python.exe")
            if os.path.isfile(candidate) and version_num > best_version_num:
                best_version_num = version_num
                best_candidate = candidate
        if best_candidate:
            return best_candidate

    # Strategy 5: Use shutil.which as last resort
    which_python = shutil.which("python")
    if which_python:
        return which_python

    return sys.executable


def _create_venv_with_env_builder(venv_dir: str) -> bool:
    """Attempt to create a virtual environment using venv.EnvBuilder (in-process).

    .. warning::
        ``EnvBuilder`` internally uses ``sys.executable`` to copy the Python
        binary into the venv.  On QGIS Windows ``sys.executable`` is
        ``qgis-bin.exe``, so this would copy QGIS itself and later subprocess
        calls would launch a new QGIS instance.  Therefore this function is
        **skipped** when ``sys.executable`` does not look like a Python
        interpreter.

    Args:
        venv_dir: Path where the venv should be created.

    Returns:
        True if the venv was created and the Python executable exists.
    """
    # Guard: only safe when sys.executable is actually Python.
    exe_name = os.path.basename(sys.executable).lower()
    if exe_name not in ("python.exe", "python3.exe", "python", "python3"):
        return False

    try:
        import venv as venv_mod

        builder = venv_mod.EnvBuilder(with_pip=True)
        builder.create(venv_dir)
        return os.path.isfile(get_venv_python_path(venv_dir))
    except Exception:
        return False


def _try_copy_python_executable(venv_dir: str) -> bool:
    """Copy the current Python executable into the venv as a recovery step.

    This handles the case where venv creation produced the directory structure
    but did not place the Python executable (known to happen with QGIS's
    embedded Python on Windows).

    Args:
        venv_dir: Path to the venv directory.

    Returns:
        True if the Python executable now exists at the expected path.
    """
    python_path = get_venv_python_path(venv_dir)
    if os.path.isfile(python_path):
        return True

    target_dir = os.path.dirname(python_path)
    os.makedirs(target_dir, exist_ok=True)

    try:
        shutil.copy2(_find_python_executable(), python_path)
        return os.path.isfile(python_path)
    except (OSError, shutil.SameFileError):
        return False


def _cleanup_partial_venv(venv_dir: str) -> None:
    """Remove a partially created venv directory (best-effort).

    Args:
        venv_dir: Path to the venv directory to clean up.
    """
    if os.path.isdir(venv_dir):
        try:
            shutil.rmtree(venv_dir)
        except OSError:
            pass


def _verify_pip_and_return(python_path: str) -> str:
    """Ensure pip is available in the venv and return the python path.

    Args:
        python_path: Path to the venv's Python executable.

    Returns:
        The *python_path* if pip is verified.

    Raises:
        RuntimeError: If pip cannot be made available.
    """
    env = _get_clean_env()
    kwargs = _get_subprocess_kwargs()

    # Try ensurepip (may already be present from EnvBuilder)
    subprocess.run(  # nosec B603
        [python_path, "-m", "ensurepip", "--upgrade"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        **kwargs,
    )

    # Verify pip works
    result = subprocess.run(  # nosec B603
        [python_path, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        **kwargs,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "pip is not available in the virtual environment.\n"
            f"Python path: {python_path}\n"
            f"Error: {result.stderr or result.stdout}"
        )

    return python_path


def create_venv(venv_dir: str) -> str:
    """Create a virtual environment at the specified path.

    When uv is available, uses ``uv venv`` which is faster and does not
    require pip to be bootstrapped inside the venv.

    Otherwise uses a multi-strategy approach to handle embedded Python
    environments (e.g. QGIS on Windows) where ``sys.executable`` may point
    to ``qgis-bin.exe`` rather than ``python.exe``.

    Pip fallback strategy order:
        1. Subprocess using the real Python executable found by
           ``_find_python_executable()`` (primary, works on Windows QGIS).
        2. In-process ``venv.EnvBuilder`` (fallback, only when
           ``sys.executable`` is already a Python interpreter).
        3. Recovery: copy the real Python executable into the venv when the
           directory was created but the executable is missing.

    Args:
        venv_dir: Path where the venv should be created.

    Returns:
        Path to the Python executable inside the newly created venv.

    Raises:
        RuntimeError: If venv creation fails after all strategies.
    """
    from .uv_manager import get_uv_path

    if not python_runtime_supported():
        raise RuntimeError(python_runtime_error())

    os.makedirs(os.path.dirname(venv_dir), exist_ok=True)

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env()
    kwargs = _get_subprocess_kwargs()

    # Strategy 0: Use uv venv when available (fastest, no pip needed)
    if _uv_usable():
        uv_path = get_uv_path()
        python_exe = _find_python_executable()
        cmd = [uv_path, "venv", "--python", python_exe, venv_dir]
        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            **kwargs,
        )
        if result.returncode == 0 and os.path.isfile(python_path):
            return python_path
        # uv venv failed — clean up and fall through to pip strategies
        _cleanup_partial_venv(venv_dir)

    # Strategy 1: Subprocess with the real Python executable
    python_exe = _find_python_executable()
    subprocess_error = ""

    cmd = [python_exe, "-m", "venv", venv_dir]
    result = subprocess.run(  # nosec B603
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        **kwargs,
    )

    if result.returncode == 0 and os.path.isfile(python_path):
        return _verify_pip_and_return(python_path)

    if result.returncode != 0:
        subprocess_error = result.stderr or result.stdout or ""

    # Clean up partial venv before retrying
    _cleanup_partial_venv(venv_dir)

    # Strategy 2: In-process EnvBuilder (skipped when sys.executable is not Python)
    if _create_venv_with_env_builder(venv_dir):
        return _verify_pip_and_return(python_path)

    _cleanup_partial_venv(venv_dir)

    # Strategy 3: Create venv without pip, then copy Python executable if needed
    strategy3_error = ""
    try:
        result2 = subprocess.run(  # nosec B603
            [python_exe, "-m", "venv", "--without-pip", venv_dir],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            **kwargs,
        )
        if result2.returncode == 0:
            if not os.path.isfile(python_path):
                _try_copy_python_executable(venv_dir)
            if os.path.isfile(python_path):
                return _verify_pip_and_return(python_path)
        else:
            strategy3_error = result2.stderr or result2.stdout or ""
    except Exception as exc:
        strategy3_error = f"{type(exc).__name__}: {exc}"

    # All strategies failed
    details = [
        f"sys.executable: {sys.executable}",
        f"Python found: {python_exe}",
        f"Target venv: {venv_dir}",
        f"Expected python: {python_path}",
        f"Platform: {sys.platform}",
    ]
    if subprocess_error:
        details.append(f"Subprocess error: {subprocess_error}")
    if strategy3_error:
        details.append(f"Strategy 3 error: {strategy3_error}")

    raise RuntimeError(
        "Failed to create virtual environment after trying multiple strategies.\n\n"
        "This can happen when QGIS bundles Python in a way that prevents\n"
        "standard venv creation.\n\n"
        "You can try installing manually with:\n"
        '  pip install "GeoAgent[providers]>=1.4.1"\n\n'
        "Details:\n" + "\n".join(f"  {d}" for d in details)
    )


def install_packages(
    venv_dir: str,
    packages: List[str],
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[bool, str]:
    """Install packages into the virtual environment.

    Uses uv when available for significantly faster installation,
    falling back to pip otherwise.

    Args:
        venv_dir: Path to the venv directory.
        packages: List of pip package names to install.
        progress_callback: Optional callback for progress updates (percent, message).

    Returns:
        Tuple of (success, message).
    """
    from .uv_manager import get_uv_path

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env()
    kwargs = _get_subprocess_kwargs()

    use_uv = _uv_usable()
    if use_uv:
        uv_path = get_uv_path()
        cmd = [
            uv_path,
            "pip",
            "install",
            "--python",
            python_path,
            "--upgrade",
        ] + packages
    else:
        cmd = [
            python_path,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--disable-pip-version-check",
            "--prefer-binary",
        ] + packages

    if progress_callback:
        installer = "uv" if use_uv else "pip"
        progress_callback(20, f"Installing ({installer}): {', '.join(packages)}...")

    result = subprocess.run(  # nosec B603
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        **kwargs,
    )

    if result.returncode != 0:
        error_output = result.stderr or result.stdout or "Unknown error"
        # Truncate long error messages
        if len(error_output) > 1000:
            error_output = "..." + error_output[-1000:]
        installer = "uv pip" if use_uv else "pip"
        return False, f"{installer} install failed:\n{error_output}"

    return True, "Packages installed successfully."


class DepsInstallWorker(QThread):
    """Worker thread for creating a venv and installing dependencies."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, group_name: str = "Core Providers", parent=None):
        super().__init__(parent)
        self.group_name = group_name or "Core Providers"

    def run(self):
        """Execute uv download, venv creation, and dependency installation."""
        try:
            from .uv_manager import download_uv

            if not python_runtime_supported():
                self.finished.emit(False, python_runtime_error())
                return

            start_time = time.time()
            venv_dir = get_venv_dir()

            # Step 0: Download uv if needed (fast package installer)
            if not _uv_usable():
                self.progress.emit(2, "Downloading uv package installer...")
                success, msg = download_uv(
                    progress_callback=lambda p, m: self.progress.emit(
                        2 + int(p * 0.03), m
                    ),
                )
                if not success:
                    # Non-fatal: fall back to pip
                    self.progress.emit(5, "uv unavailable, using pip instead.")
                else:
                    self.progress.emit(5, "uv ready.")

            # Step 1: Create venv if needed
            if not venv_exists():
                self.progress.emit(5, "Creating virtual environment...")
                try:
                    create_venv(venv_dir)
                except RuntimeError as e:
                    self.finished.emit(False, str(e))
                    return
            self.progress.emit(10, "Virtual environment ready.")

            # Step 2: Verify pip (only needed when not using uv)
            use_uv = _uv_usable()
            if not use_uv:
                self.progress.emit(12, "Verifying pip...")
                python_path = get_venv_python_path(venv_dir)
                env = _get_clean_env()
                kwargs = _get_subprocess_kwargs()

                result = subprocess.run(  # nosec B603
                    [python_path, "-m", "pip", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env,
                    **kwargs,
                )
                if result.returncode != 0:
                    self.finished.emit(
                        False,
                        "pip is not available in the virtual environment.\n"
                        "Please install dependencies manually:\n"
                        'pip install "GeoAgent[providers]>=1.4.1"',
                    )
                    return
            self.progress.emit(15, "Package installer ready.")

            # Step 3: Install missing packages
            missing = get_missing_packages(self.group_name)
            if not missing:
                self.finished.emit(
                    True,
                    f"All {self.group_name} dependencies are already installed.",
                )
                return

            self.progress.emit(20, f"Installing: {', '.join(missing)}...")
            success, message = install_packages(
                venv_dir,
                missing,
                progress_callback=lambda p, m: self.progress.emit(
                    20 + int(p * 0.65), m
                ),
            )
            if not success:
                self.finished.emit(False, message)
                return
            self.progress.emit(85, "Packages installed.")

            # Step 4: Add venv to sys.path
            self.progress.emit(90, "Configuring package paths...")
            ensure_venv_packages_available()

            # Step 5: Verify imports
            self.progress.emit(95, "Verifying installations...")
            still_missing = get_missing_packages(self.group_name)

            elapsed = time.time() - start_time
            if elapsed >= 60:
                minutes, seconds = divmod(int(round(elapsed)), 60)
                elapsed_str = f"{minutes}:{seconds:02d}"
            else:
                elapsed_str = f"{elapsed:.1f}s"

            if still_missing:
                self.finished.emit(
                    False,
                    f"The following packages could not be verified: "
                    f"{', '.join(still_missing)}.\n"
                    "You may need to restart QGIS for changes to take effect.",
                )
            else:
                self.progress.emit(100, f"All dependencies installed in {elapsed_str}!")
                self.finished.emit(
                    True,
                    f"All {self.group_name} dependencies installed in {elapsed_str}!",
                )

        except subprocess.TimeoutExpired:
            self.finished.emit(False, "Installation timed out after 10 minutes.")
        except Exception as e:
            self.finished.emit(False, f"Unexpected error: {str(e)}")
