"""
UV Package Installer Manager for OpenGeoAgent.

Downloads and manages the uv package installer binary for fast
dependency installation in the plugin's virtual environment.

All ``subprocess`` calls in this module use list-form argv built from
internal constants (the resolved ``uv`` binary path, fixed flags) and
never accept user-supplied input or run with ``shell=True``. The
``# nosec`` annotations on the import and call sites document this
explicitly for the plugins.qgis.org Bandit scan.

Source: https://github.com/astral-sh/uv
"""

import os
import sys
import platform
import stat
import subprocess  # nosec B404
import tarfile
import zipfile
import tempfile
import shutil
from typing import Callable, Optional, Tuple

from qgis.core import QgsMessageLog, Qgis, QgsBlockingNetworkRequest
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".open_geoagent")
UV_DIR = os.path.join(CACHE_DIR, "uv")

# Pin a known-good uv version.
# Verified platforms: x86_64-pc-windows-msvc, x86_64-apple-darwin,
# aarch64-apple-darwin, x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu.
UV_VERSION = "0.10.6"


def _log(message, level=Qgis.MessageLevel.Info):
    """Log a message to the QGIS message log.

    Args:
        message: The message to log.
        level: The log level (Qgis.MessageLevel.Info, Qgis.MessageLevel.Warning,
            Qgis.MessageLevel.Critical).
    """
    QgsMessageLog.logMessage(str(message), "OpenGeoAgent", level=level)


def get_uv_path() -> str:
    """Get the path to the uv binary.

    Returns:
        The absolute path to the uv executable.
    """
    if sys.platform == "win32":
        return os.path.join(UV_DIR, "uv.exe")
    return os.path.join(UV_DIR, "uv")


def uv_exists() -> bool:
    """Check if the uv binary is already installed.

    Returns:
        True if the uv executable exists.
    """
    return os.path.exists(get_uv_path())


def _get_uv_platform_info() -> Tuple[str, str]:
    """Get platform and architecture info for the uv download URL.

    Returns:
        A tuple of (platform_string, file_extension).
    """
    system = sys.platform
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return ("aarch64-apple-darwin", ".tar.gz")
        return ("x86_64-apple-darwin", ".tar.gz")
    elif system == "win32":
        return ("x86_64-pc-windows-msvc", ".zip")
    else:
        if machine in ("arm64", "aarch64"):
            return ("aarch64-unknown-linux-gnu", ".tar.gz")
        return ("x86_64-unknown-linux-gnu", ".tar.gz")


def get_uv_download_url() -> str:
    """Construct the download URL for the uv binary.

    Returns:
        The full download URL string.
    """
    platform_str, ext = _get_uv_platform_info()
    filename = f"uv-{platform_str}{ext}"
    return (
        f"https://github.com/astral-sh/uv/releases/download/" f"{UV_VERSION}/{filename}"
    )


def _safe_extract_tar(tar, dest_dir):
    """Safely extract tar archive with path traversal protection.

    Args:
        tar: An open tarfile.TarFile object.
        dest_dir: Destination directory for extraction.
    """
    dest_dir = os.path.realpath(dest_dir)
    use_filter = sys.version_info >= (3, 12)
    for member in tar.getmembers():
        member_path = os.path.realpath(os.path.join(dest_dir, member.name))
        if not member_path.startswith(dest_dir + os.sep) and member_path != dest_dir:
            raise ValueError(f"Attempted path traversal in tar archive: {member.name}")
        if use_filter:
            tar.extract(member, dest_dir, filter="data")
        else:
            tar.extract(member, dest_dir)


def _safe_extract_zip(zip_file, dest_dir):
    """Safely extract zip archive with path traversal protection.

    Args:
        zip_file: An open zipfile.ZipFile object.
        dest_dir: Destination directory for extraction.
    """
    dest_dir = os.path.realpath(dest_dir)
    for member in zip_file.namelist():
        member_path = os.path.realpath(os.path.join(dest_dir, member))
        if not member_path.startswith(dest_dir + os.sep) and member_path != dest_dir:
            raise ValueError(f"Attempted path traversal in zip archive: {member}")
        zip_file.extract(member, dest_dir)


def _find_file_in_dir(directory, filename):
    """Find a file by name within a directory tree.

    Args:
        directory: The root directory to search.
        filename: The filename to find.

    Returns:
        The full path to the file, or None if not found.
    """
    for root, _dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def download_uv(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    """Download and install the uv binary using QGIS network manager.

    Uses QgsBlockingNetworkRequest to respect QGIS proxy settings.

    Args:
        progress_callback: Function called with (percent, message) for progress.
        cancel_check: Function that returns True if operation should be cancelled.

    Returns:
        A tuple of (success: bool, message: str).
    """
    if uv_exists():
        _log("uv already exists")
        return True, "uv already installed"

    url = get_uv_download_url()
    _log(f"Downloading uv {UV_VERSION} from: {url}")

    if progress_callback:
        progress_callback(0, f"Downloading uv {UV_VERSION}...")

    _, ext = _get_uv_platform_info()
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)

    try:
        if cancel_check and cancel_check():
            return False, "Download cancelled"

        request = QgsBlockingNetworkRequest()
        qurl = QUrl(url)

        if progress_callback:
            progress_callback(5, "Connecting to download server...")

        err = request.get(QNetworkRequest(qurl))

        if err != QgsBlockingNetworkRequest.NoError:
            error_msg = request.errorMessage()
            if "404" in error_msg or "Not Found" in error_msg:
                error_msg = (
                    f"uv {UV_VERSION} not available for this platform. " f"URL: {url}"
                )
            else:
                error_msg = f"Download failed: {error_msg}"
            _log(error_msg, Qgis.MessageLevel.Critical)
            return False, error_msg

        if cancel_check and cancel_check():
            return False, "Download cancelled"

        reply = request.reply()
        content = reply.content()

        if progress_callback:
            total_mb = len(content) / (1024 * 1024)
            progress_callback(50, f"Downloaded {total_mb:.1f} MB, saving...")

        with open(temp_path, "wb") as f:
            f.write(content.data())

        _log(f"Download complete ({len(content)} bytes), extracting...")

        if progress_callback:
            progress_callback(60, "Extracting uv...")

        if os.path.exists(UV_DIR):
            shutil.rmtree(UV_DIR)
        os.makedirs(UV_DIR, exist_ok=True)

        # Extract archive to a temporary directory, then move the binary
        extract_dir = tempfile.mkdtemp()
        try:
            if temp_path.endswith(".zip"):
                with zipfile.ZipFile(temp_path, "r") as z:
                    _safe_extract_zip(z, extract_dir)
            else:
                with tarfile.open(temp_path, "r:gz") as tar:
                    _safe_extract_tar(tar, extract_dir)

            # Find the uv binary in the extracted contents
            uv_binary_name = "uv.exe" if sys.platform == "win32" else "uv"
            uv_binary = _find_file_in_dir(extract_dir, uv_binary_name)

            if uv_binary is None:
                found_files = []
                for root, _dirs, files in os.walk(extract_dir):
                    found_files.extend(files)
                    if len(found_files) >= 20:
                        break
                files_str = ", ".join(found_files[:20])
                return False, (
                    f"uv binary not found in archive. "
                    f"Expected '{uv_binary_name}'. "
                    f"Files found (up to 20): {files_str}"
                )

            dest = get_uv_path()
            shutil.copy2(uv_binary, dest)

            # Set executable permissions on Unix
            if sys.platform != "win32":
                os.chmod(
                    dest,
                    stat.S_IRWXU
                    | stat.S_IRGRP
                    | stat.S_IXGRP
                    | stat.S_IROTH
                    | stat.S_IXOTH,
                )

        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

        if progress_callback:
            progress_callback(80, "Verifying uv installation...")

        success, verify_msg = verify_uv()

        if success:
            if progress_callback:
                progress_callback(100, f"uv {UV_VERSION} installed")
            _log("uv installed successfully")
            return True, f"uv {UV_VERSION} installed successfully"
        else:
            return False, f"Verification failed: {verify_msg}"

    except InterruptedError:
        return False, "Download cancelled"
    except Exception as e:
        error_msg = f"uv installation failed: {str(e)}"
        _log(error_msg, Qgis.MessageLevel.Critical)
        return False, error_msg
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def verify_uv() -> Tuple[bool, str]:
    """Verify that the uv binary works.

    Returns:
        A tuple of (success: bool, message: str).
    """
    uv_path = get_uv_path()

    if not os.path.exists(uv_path):
        return False, f"uv not found at {uv_path}"

    try:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)

        kwargs = {}
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = startupinfo

        result = subprocess.run(  # nosec B603
            [uv_path, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            **kwargs,
        )

        if result.returncode == 0:
            version_output = result.stdout.strip()
            _log(f"Verified uv: {version_output}")
            return True, version_output
        else:
            error = result.stderr or "Unknown error"
            _log(f"uv verification failed: {error}", Qgis.MessageLevel.Warning)
            return False, f"Verification failed: {error[:100]}"

    except subprocess.TimeoutExpired:
        return False, "uv verification timed out"
    except Exception as e:
        return False, f"Verification error: {str(e)[:100]}"


def remove_uv() -> Tuple[bool, str]:
    """Remove the uv installation.

    Returns:
        A tuple of (success: bool, message: str).
    """
    if not os.path.exists(UV_DIR):
        return True, "uv not installed"

    try:
        shutil.rmtree(UV_DIR)
        _log("Removed uv installation")
        return True, "uv removed"
    except Exception as e:
        error_msg = f"Failed to remove uv: {str(e)}"
        _log(error_msg, Qgis.MessageLevel.Warning)
        return False, error_msg
