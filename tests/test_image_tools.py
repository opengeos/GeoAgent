"""Tests for GeoAgent image generation tools."""

from __future__ import annotations

import base64
import sys
import types

from geoagent.tools.images import image_generation_tools


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
        def __init__(self) -> None:
            self.images = _Images()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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
    filename = result["images"][0]["path"].split("/")[-1]
    assert (tmp_path / filename).read_bytes() == b"fake-png"


def test_generate_image_reports_missing_api_key(monkeypatch) -> None:
    """Verify image generation gives a clear setup error without API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

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
        def __init__(self) -> None:
            self.images = _Images()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))

    tool = {item.tool_name: item for item in image_generation_tools()}["generate_image"]
    result = tool.__wrapped__("digital globe", output_dir=str(tmp_path))

    assert result["success"] is True
    assert result["requested_model"] == "gpt-image-2"
    assert result["model"] == "gpt-image-1"
    assert "not verified" in result["fallback_reason"]
    assert models == ["gpt-image-2", "gpt-image-1"]
