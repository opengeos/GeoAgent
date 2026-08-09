"""Tests for GeoAgent image generation tools."""

from __future__ import annotations

import base64
import io
import json
import sys
import types
import urllib.error
import urllib.request

from geoagent.tools.images import image_generation_tools


def _get_generate_image():
    """Return the ``generate_image`` tool callable."""
    tools = {item.tool_name: item for item in image_generation_tools()}
    return tools["generate_image"]


def _use_minimax_env(monkeypatch):
    """Configure a hermetic MiniMax-only environment for a test."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    for name in (
        "MINIMAX_API_REGION",
        "MINIMAX_IMAGE_ENDPOINT",
        "MINIMAX_IMAGE_RESPONSE_FORMAT",
        "GEOAGENT_IMAGE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def _stub_minimax_urlopen(monkeypatch, response_body, captured):
    """Patch ``urlopen`` to capture the request and return ``response_body``."""

    class _Response:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def read(self) -> bytes:
            return self._body

    def _fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response(json.dumps(response_body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)


def test_generate_image_uses_minimax_url_response(monkeypatch) -> None:
    """Verify MiniMax URL responses return image links from image_urls."""
    captured: dict = {}
    body = {
        "id": "trace-id",
        "data": {"image_urls": ["https://example.com/a.png"]},
        "metadata": {"success_count": "1", "failed_count": "0"},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    _use_minimax_env(monkeypatch)
    _stub_minimax_urlopen(monkeypatch, body, captured)

    result = _get_generate_image().__wrapped__(
        "a mountain landscape",
        aspect_ratio="16:9",
        prompt_optimizer=True,
    )

    assert result["success"] is True
    assert result["model"] == "image-01"
    assert result["images"][0]["url"] == "https://example.com/a.png"
    assert result["success_count"] == "1"
    assert captured["url"] == "https://api.minimax.io/v1/image_generation"
    assert captured["payload"]["model"] == "image-01"
    assert captured["payload"]["prompt"] == "a mountain landscape"
    assert captured["payload"]["aspect_ratio"] == "16:9"
    assert captured["payload"]["prompt_optimizer"] is True
    assert captured["payload"]["response_format"] == "url"
    header_keys = {key.lower(): value for key, value in captured["headers"].items()}
    assert header_keys["authorization"] == "Bearer test-key"


def test_generate_image_writes_minimax_base64_bytes(monkeypatch, tmp_path) -> None:
    """Verify MiniMax base64 responses are decoded and written to files."""
    captured: dict = {}
    image_bytes = b"\x89PNG\r\n\x1a\nminimax"
    body = {
        "data": {"image_base64": [base64.b64encode(image_bytes).decode("ascii")]},
        "metadata": {"success_count": 1, "failed_count": 0},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    _use_minimax_env(monkeypatch)
    _stub_minimax_urlopen(monkeypatch, body, captured)

    result = _get_generate_image().__wrapped__(
        "orange tabby cat",
        output_dir=str(tmp_path),
        response_format="base64",
        model="image-01-live",
    )

    assert result["success"] is True
    assert result["model"] == "image-01-live"
    path = result["images"][0]["path"]
    assert path.endswith(".png")
    assert result["images"][0]["mime_type"] == "image/png"
    filename = path.split("/")[-1]
    assert (tmp_path / filename).read_bytes() == image_bytes
    assert captured["payload"]["response_format"] == "base64"


def test_generate_image_minimax_base64_defaults_to_jpeg(monkeypatch, tmp_path) -> None:
    """Verify MiniMax base64 bytes are saved with their sniffed image format."""
    captured: dict = {}
    image_bytes = b"\xff\xd8\xff\xe0minimax"
    body = {
        "data": {"image_base64": [base64.b64encode(image_bytes).decode("ascii")]},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    _use_minimax_env(monkeypatch)
    _stub_minimax_urlopen(monkeypatch, body, captured)

    result = _get_generate_image().__wrapped__(
        "orange tabby cat",
        output_dir=str(tmp_path),
        response_format="base64",
    )

    assert result["success"] is True
    assert result["images"][0]["path"].endswith(".jpg")
    assert result["images"][0]["mime_type"] == "image/jpeg"


def test_generate_image_minimax_cn_region_endpoint(monkeypatch) -> None:
    """Verify the CN region selects the minimaxi endpoint host."""
    captured: dict = {}
    body = {
        "data": {"image_urls": ["https://example.com/b.png"]},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    _use_minimax_env(monkeypatch)
    monkeypatch.setenv("MINIMAX_API_REGION", "cn_zh")
    _stub_minimax_urlopen(monkeypatch, body, captured)

    result = _get_generate_image().__wrapped__("a river delta")

    assert result["success"] is True
    assert captured["url"] == "https://api.minimaxi.com/v1/image_generation"


def test_generate_image_minimax_rejects_non_http_endpoint(monkeypatch) -> None:
    """Verify a non-http endpoint override falls back to the region default."""
    captured: dict = {}
    body = {
        "data": {"image_urls": ["https://example.com/c.png"]},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    _use_minimax_env(monkeypatch)
    monkeypatch.setenv("MINIMAX_IMAGE_ENDPOINT", "file:///etc/passwd")
    _stub_minimax_urlopen(monkeypatch, body, captured)

    result = _get_generate_image().__wrapped__("a salt flat")

    assert result["success"] is True
    assert captured["url"] == "https://api.minimax.io/v1/image_generation"


def test_generate_image_minimax_reports_status_error(monkeypatch) -> None:
    """Verify a non-zero base_resp status_code is surfaced as an error."""
    captured: dict = {}
    body = {
        "data": {},
        "base_resp": {"status_code": 1004, "status_msg": "authentication failed"},
    }
    _use_minimax_env(monkeypatch)
    _stub_minimax_urlopen(monkeypatch, body, captured)

    result = _get_generate_image().__wrapped__("a coastline")

    assert result["success"] is False
    assert result["status_code"] == 1004
    assert "authentication failed" in result["error"]


def test_generate_image_minimax_surfaces_http_error_body(monkeypatch) -> None:
    """Verify the MiniMax HTTP error payload is included in the error text."""

    def _fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b'{"base_resp":{"status_msg":"rate limit reached"}}'),
        )

    _use_minimax_env(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    result = _get_generate_image().__wrapped__("a glacier")

    assert result["success"] is False
    assert "HTTP 429" in result["error"]
    assert "rate limit reached" in result["error"]


def test_generate_image_explicit_openai_model_bypasses_minimax(
    monkeypatch, tmp_path
) -> None:
    """Verify an explicit OpenAI model still reaches OpenAI when MiniMax is set."""
    image_payload = base64.b64encode(b"fake-png").decode("ascii")
    calls: dict = {}

    class _Images:
        def generate(self, **kwargs):
            calls.update(kwargs)
            item = types.SimpleNamespace(
                b64_json=image_payload,
                revised_prompt="",
                url=None,
            )
            return types.SimpleNamespace(data=[item])

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.images = _Images()

    _use_minimax_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))

    result = _get_generate_image().__wrapped__(
        "a fjord",
        model="gpt-image-1",
        output_dir=str(tmp_path),
    )

    assert result["success"] is True
    assert calls["model"] == "gpt-image-1"


def test_generate_image_writes_openai_image_bytes(monkeypatch, tmp_path) -> None:
    """Verify generate_image saves OpenAI b64 image responses as files."""
    image_payload = base64.b64encode(b"fake-png").decode("ascii")
    calls = {}

    class _Images:
        def generate(self, **kwargs):
            calls.update(kwargs)
            item = types.SimpleNamespace(
                b64_json=image_payload,
                revised_prompt="revised prompt",
                url=None,
            )
            return types.SimpleNamespace(data=[item])

    class _Client:
        def __init__(self, **kwargs) -> None:
            calls["timeout"] = kwargs.get("timeout")
            self.images = _Images()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))

    tool = {item.tool_name: item for item in image_generation_tools()}["generate_image"]
    result = tool.__wrapped__(
        "orange tabby cat",
        output_dir=str(tmp_path),
        quality="low",
    )

    assert result["success"] is True
    assert result["images"][0]["path"].endswith(".png")
    assert result["images"][0]["revised_prompt"] == "revised prompt"
    assert calls["model"] == "gpt-image-2"
    assert calls["prompt"] == "orange tabby cat"
    assert calls["timeout"] == 180.0
    filename = result["images"][0]["path"].split("/")[-1]
    assert (tmp_path / filename).read_bytes() == b"fake-png"


def test_generate_image_reports_missing_api_key(monkeypatch) -> None:
    """Verify image generation gives a clear setup error without API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    tool = {item.tool_name: item for item in image_generation_tools()}["generate_image"]
    result = tool.__wrapped__("orange tabby cat")

    assert result["success"] is False
    assert "OPENAI_API_KEY" in result["error"]


def test_generate_image_falls_back_on_unverified_gpt_image_2(
    monkeypatch, tmp_path
) -> None:
    """Verify gpt-image-2 permission failures retry with gpt-image-1."""
    image_payload = base64.b64encode(b"fallback-png").decode("ascii")
    models = []

    class _Images:
        def generate(self, **kwargs):
            models.append(kwargs["model"])
            if kwargs["model"] == "gpt-image-2":
                raise RuntimeError(
                    "organization is not verified for gpt-image-2 "
                    "(403 PermissionDeniedError)"
                )
            item = types.SimpleNamespace(
                b64_json=image_payload,
                revised_prompt="fallback prompt",
                url=None,
            )
            return types.SimpleNamespace(data=[item])

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.images = _Images()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))

    tool = {item.tool_name: item for item in image_generation_tools()}["generate_image"]
    result = tool.__wrapped__("digital globe", output_dir=str(tmp_path))

    assert result["success"] is True
    assert result["requested_model"] == "gpt-image-2"
    assert result["model"] == "gpt-image-1"
    assert "not verified" in result["fallback_reason"]
    assert models == ["gpt-image-2", "gpt-image-1"]


def test_generate_image_uses_configured_model_and_timeout(
    monkeypatch, tmp_path
) -> None:
    """Verify QGIS image settings can steer direct tool defaults."""
    image_payload = base64.b64encode(b"configured-png").decode("ascii")
    calls = {}

    class _Images:
        def generate(self, **kwargs):
            calls.update(kwargs)
            item = types.SimpleNamespace(
                b64_json=image_payload,
                revised_prompt="configured prompt",
                url=None,
            )
            return types.SimpleNamespace(data=[item])

    class _Client:
        def __init__(self, **kwargs) -> None:
            calls["timeout"] = kwargs.get("timeout")
            self.images = _Images()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("GEOAGENT_IMAGE_MODEL", "gpt-image-1")
    monkeypatch.setenv("GEOAGENT_IMAGE_TIMEOUT", "5")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))

    tool = {item.tool_name: item for item in image_generation_tools()}["generate_image"]
    result = tool.__wrapped__("satellite image of seattle", output_dir=str(tmp_path))

    assert result["success"] is True
    assert result["model"] == "gpt-image-1"
    assert result["timeout"] == 5.0
    assert calls["model"] == "gpt-image-1"
    assert calls["timeout"] == 5.0
