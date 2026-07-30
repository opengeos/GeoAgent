"""Image generation tools for GeoAgent."""

from __future__ import annotations

import base64
import binascii
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from geoagent.core.decorators import geo_tool

DEFAULT_IMAGE_MODEL = "gpt-image-2"
FALLBACK_IMAGE_MODEL = "gpt-image-1"
DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_IMAGE_QUALITY = "low"
DEFAULT_IMAGE_TIMEOUT = 180.0
SUPPORTED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}
SUPPORTED_IMAGE_QUALITIES = {"low", "medium", "high", "auto"}

MINIMAX_DEFAULT_IMAGE_MODEL = "image-01"
MINIMAX_IMAGE_MODELS = {"image-01", "image-01-live"}
MINIMAX_IMAGE_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/image_generation",
    "cn_zh": "https://api.minimaxi.com/v1/image_generation",
}
MINIMAX_DEFAULT_REGION = "global_en"
MINIMAX_ASPECT_RATIOS = {
    "1:1",
    "16:9",
    "4:3",
    "3:2",
    "2:3",
    "3:4",
    "9:16",
    "21:9",
}
MINIMAX_RESPONSE_FORMATS = {"url", "base64"}
MINIMAX_DEFAULT_RESPONSE_FORMAT = "url"
MINIMAX_MIN_DIMENSION = 512
MINIMAX_MAX_DIMENSION = 2048
MINIMAX_MAX_IMAGES = 9


def _output_dir(output_dir: str | None = None) -> Path:
    """Return the directory used for generated image files."""
    path = Path(
        output_dir or os.environ.get("GEOAGENT_IMAGE_OUTPUT_DIR", "")
    ).expanduser()
    if not str(path).strip() or str(path) == ".":
        path = Path(tempfile.gettempdir()) / "geoagent_images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_image_stem(value: str | None = None) -> str:
    """Return a compact filesystem-safe image filename stem."""
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").lower())
    text = "_".join(part for part in text.split("_") if part)
    if not text:
        text = "geoagent_image"
    return text[:60]


def _response_data_items(response: Any) -> list[Any]:
    """Return image data items from an OpenAI response object or dict."""
    if isinstance(response, dict):
        data = response.get("data", [])
    else:
        data = getattr(response, "data", [])
    return list(data or [])


def _item_value(item: Any, key: str) -> Any:
    """Read a field from a response item object or dict."""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _is_image_model_permission_error(exc: Exception) -> bool:
    """Return True when an OpenAI image model is unavailable for the org."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "permissiondenied",
            "permission denied",
            "not verified",
            "organization is not verified",
            "403",
        )
    )


def _image_timeout() -> float:
    """Return the image-generation request timeout in seconds."""
    raw = os.environ.get("GEOAGENT_IMAGE_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_IMAGE_TIMEOUT
    try:
        timeout = float(raw)
    except ValueError:
        return DEFAULT_IMAGE_TIMEOUT
    return timeout if timeout > 0 else DEFAULT_IMAGE_TIMEOUT


def _generate_openai_image(
    client: Any,
    *,
    model: str,
    prompt: str,
    size: str,
    quality: str,
) -> Any:
    """Call the OpenAI image generation endpoint."""
    return client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
    )


def _minimax_image_enabled() -> bool:
    """Return True when a MiniMax API key is configured."""
    return bool(os.environ.get("MINIMAX_API_KEY", "").strip())


def _minimax_image_endpoint() -> str:
    """Return the MiniMax image generation endpoint for the active region."""
    explicit = os.environ.get("MINIMAX_IMAGE_ENDPOINT", "").strip()
    if explicit:
        return explicit
    region = os.environ.get("MINIMAX_API_REGION", "").strip().lower()
    return MINIMAX_IMAGE_ENDPOINTS.get(
        region, MINIMAX_IMAGE_ENDPOINTS[MINIMAX_DEFAULT_REGION]
    )


def _minimax_image_model(model: str) -> str:
    """Resolve the MiniMax image model from the argument or environment."""
    requested = str(model or "").strip()
    if requested in MINIMAX_IMAGE_MODELS:
        return requested
    configured = os.environ.get("GEOAGENT_IMAGE_MODEL", "").strip()
    if configured in MINIMAX_IMAGE_MODELS:
        return configured
    return MINIMAX_DEFAULT_IMAGE_MODEL


def _minimax_response_format(response_format: str) -> str:
    """Resolve the MiniMax response format (``url`` or ``base64``)."""
    requested = str(response_format or "").strip().lower()
    if requested in MINIMAX_RESPONSE_FORMATS:
        return requested
    configured = os.environ.get("MINIMAX_IMAGE_RESPONSE_FORMAT", "").strip().lower()
    if configured in MINIMAX_RESPONSE_FORMATS:
        return configured
    return MINIMAX_DEFAULT_RESPONSE_FORMAT


def _minimax_dimensions(size: str) -> tuple[int, int] | None:
    """Parse a ``WIDTHxHEIGHT`` string into MiniMax-compatible dimensions."""
    text = str(size or "").strip().lower()
    if "x" not in text:
        return None
    width_text, _, height_text = text.partition("x")
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError:
        return None
    for value in (width, height):
        if (
            value < MINIMAX_MIN_DIMENSION
            or value > MINIMAX_MAX_DIMENSION
            or value % 8 != 0
        ):
            return None
    return width, height


def _minimax_status_code(base_resp: Any) -> Any:
    """Read ``base_resp.status_code`` from the MiniMax response."""
    if isinstance(base_resp, dict):
        return base_resp.get("status_code")
    return getattr(base_resp, "status_code", None)


def _post_minimax_image(
    endpoint: str, api_key: str, payload: dict[str, Any], timeout: float
) -> bytes:
    """POST a JSON payload to the MiniMax image endpoint and return the body."""
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _generate_minimax_image(
    *,
    prompt: str,
    size: str,
    model: str,
    output_dir: str,
    aspect_ratio: str,
    response_format: str,
    seed: int,
    n: int,
    prompt_optimizer: bool,
    subject_reference: str,
) -> dict[str, Any]:
    """Generate an image with the MiniMax image generation endpoint."""
    api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    endpoint = _minimax_image_endpoint()
    resolved_model = _minimax_image_model(model)
    fmt = _minimax_response_format(response_format)
    count = n if isinstance(n, int) and 1 <= n <= MINIMAX_MAX_IMAGES else 1

    payload: dict[str, Any] = {
        "model": resolved_model,
        "prompt": prompt,
        "response_format": fmt,
        "n": count,
    }
    ratio = str(aspect_ratio or "").strip()
    if ratio in MINIMAX_ASPECT_RATIOS:
        payload["aspect_ratio"] = ratio
    else:
        dimensions = _minimax_dimensions(size)
        if dimensions is not None:
            payload["width"], payload["height"] = dimensions
    if isinstance(seed, int) and seed > 0:
        payload["seed"] = seed
    if prompt_optimizer:
        payload["prompt_optimizer"] = True
    reference = str(subject_reference or "").strip()
    if reference:
        payload["subject_reference"] = [{"type": "character", "image_file": reference}]

    timeout = _image_timeout()
    try:
        raw = _post_minimax_image(endpoint, api_key, payload, timeout)
    except (urllib.error.URLError, OSError) as exc:
        return {
            "success": False,
            "error": (
                f"Image generation request failed with {resolved_model} "
                f"within {timeout:g}s: {exc}"
            ),
            "model": resolved_model,
            "timeout": timeout,
        }

    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return {
            "success": False,
            "error": f"The image API returned an unreadable response: {exc}",
            "model": resolved_model,
        }

    status_code = _minimax_status_code(data.get("base_resp"))
    if status_code not in (0, "0"):
        base_resp = data.get("base_resp") or {}
        status_msg = ""
        if isinstance(base_resp, dict):
            status_msg = str(base_resp.get("status_msg") or "")
        return {
            "success": False,
            "error": (
                f"Image generation failed (status_code {status_code})"
                + (f": {status_msg}" if status_msg else ".")
            ),
            "model": resolved_model,
            "status_code": status_code,
        }

    payload_data = data.get("data") or {}
    out_dir = _output_dir(output_dir or None)
    images: list[dict[str, Any]] = []
    decode_errors: list[str] = []
    for index, value in enumerate(payload_data.get("image_base64") or [], start=1):
        try:
            image_bytes = base64.b64decode(str(value), validate=True)
        except (binascii.Error, ValueError) as exc:
            decode_errors.append(f"item {index}: {exc}")
            continue
        stem = _safe_image_stem(prompt)
        suffix = time.strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"{stem}-{suffix}-{index}.png"
        with open(path, "wb") as f:
            f.write(image_bytes)
        images.append(
            {
                "path": str(path),
                "format": "png",
                "mime_type": "image/png",
                "revised_prompt": "",
            }
        )
    for value in payload_data.get("image_urls") or []:
        if value:
            images.append(
                {
                    "url": str(value),
                    "format": "url",
                    "mime_type": "",
                    "revised_prompt": "",
                }
            )

    if not images:
        error_message = "The image API response did not include an image."
        if decode_errors:
            error_message = (
                "The image API returned invalid base64 payload(s): "
                + "; ".join(decode_errors)
            )
        return {
            "success": False,
            "error": error_message,
            "model": resolved_model,
        }

    metadata = data.get("metadata") or {}
    return {
        "success": True,
        "prompt": prompt,
        "model": resolved_model,
        "requested_model": resolved_model,
        "response_format": fmt,
        "timeout": timeout,
        "images": images,
        "path": images[0].get("path", ""),
        "url": images[0].get("url", ""),
        "success_count": metadata.get("success_count"),
        "failed_count": metadata.get("failed_count"),
        "message": f"Generated {len(images)} image(s).",
    }


def image_generation_tools() -> list[Any]:
    """Return tools for generating standalone image files."""

    @geo_tool(
        category="image_generation",
        description=(
            "Generate an image from a text prompt. "
            "Use this when the user asks to create, draw, render, or generate "
            "a picture."
        ),
        available_in=("full", "fast"),
        requires_packages=("openai",),
    )
    def generate_image(
        prompt: str,
        size: str = DEFAULT_IMAGE_SIZE,
        quality: str = DEFAULT_IMAGE_QUALITY,
        model: str = "",
        output_dir: str = "",
        aspect_ratio: str = "",
        response_format: str = "",
        seed: int = 0,
        n: int = 1,
        prompt_optimizer: bool = False,
        subject_reference: str = "",
    ) -> dict[str, Any]:
        """Generate an image file from a text prompt.

        Args:
            prompt: Visual description of the image to generate.
            size: One of 1024x1024, 1024x1536, 1536x1024, or auto.
            quality: One of low, medium, high, or auto.
            model: Image model to use. Defaults to ``GEOAGENT_IMAGE_MODEL``
                when set, otherwise the backend default.
            output_dir: Optional directory for the generated image.
            aspect_ratio: MiniMax aspect ratio (for example ``1:1`` or
                ``16:9``). When set it takes priority over ``size``.
            response_format: MiniMax output format, ``url`` or ``base64``.
            seed: Optional MiniMax random seed for reproducible images.
            n: Number of MiniMax images to generate (1-9).
            prompt_optimizer: Enable MiniMax automatic prompt optimization.
            subject_reference: Optional MiniMax character reference image
                (URL or base64) to keep a consistent subject.

        Returns:
            A JSON-friendly result containing local image file paths and image
            metadata. The host UI can render the returned ``images`` list.
        """
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"success": False, "error": "Image prompt is empty."}
        if _minimax_image_enabled():
            return _generate_minimax_image(
                prompt=prompt,
                size=size,
                model=model,
                output_dir=output_dir,
                aspect_ratio=aspect_ratio,
                response_format=response_format,
                seed=seed,
                n=n,
                prompt_optimizer=prompt_optimizer,
                subject_reference=subject_reference,
            )
        if not os.environ.get("OPENAI_API_KEY"):
            return {
                "success": False,
                "error": (
                    "OPENAI_API_KEY is required for image generation. Add an "
                    "OpenAI API key in OpenGeoAgent Settings > Model or set "
                    "OPENAI_API_KEY."
                ),
            }

        size = str(size or DEFAULT_IMAGE_SIZE).strip()
        if size not in SUPPORTED_IMAGE_SIZES:
            size = DEFAULT_IMAGE_SIZE
        quality = str(quality or DEFAULT_IMAGE_QUALITY).strip()
        if quality not in SUPPORTED_IMAGE_QUALITIES:
            quality = DEFAULT_IMAGE_QUALITY
        configured_model = os.environ.get("GEOAGENT_IMAGE_MODEL", "").strip()
        requested_model = str(model or configured_model or DEFAULT_IMAGE_MODEL).strip()
        requested_model = requested_model or DEFAULT_IMAGE_MODEL
        model = requested_model

        from openai import OpenAI

        timeout = _image_timeout()
        client = OpenAI(timeout=timeout)
        fallback_reason = ""
        try:
            response = _generate_openai_image(
                client,
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
            )
        except Exception as exc:
            if model != FALLBACK_IMAGE_MODEL and _is_image_model_permission_error(exc):
                fallback_reason = str(exc)
                model = FALLBACK_IMAGE_MODEL
                response = _generate_openai_image(
                    client,
                    model=model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                )
            else:
                return {
                    "success": False,
                    "error": (
                        f"Image generation failed with {model} "
                        f"within {timeout:g}s: {exc}"
                    ),
                    "model": model,
                    "timeout": timeout,
                }

        out_dir = _output_dir(output_dir or None)
        images: list[dict[str, Any]] = []
        decode_errors: list[str] = []
        for index, item in enumerate(_response_data_items(response), start=1):
            b64_json = _item_value(item, "b64_json")
            url = _item_value(item, "url")
            revised_prompt = _item_value(item, "revised_prompt")
            if b64_json:
                try:
                    image_bytes = base64.b64decode(str(b64_json), validate=True)
                except (binascii.Error, ValueError) as exc:
                    decode_errors.append(f"item {index}: {exc}")
                    continue
                stem = _safe_image_stem(prompt)
                suffix = time.strftime("%Y%m%d-%H%M%S")
                path = out_dir / f"{stem}-{suffix}-{index}.png"
                with open(path, "wb") as f:
                    f.write(image_bytes)
                images.append(
                    {
                        "path": str(path),
                        "format": "png",
                        "mime_type": "image/png",
                        "revised_prompt": revised_prompt or "",
                    }
                )
            elif url:
                images.append(
                    {
                        "url": str(url),
                        "format": "url",
                        "mime_type": "",
                        "revised_prompt": revised_prompt or "",
                    }
                )

        if not images:
            error_message = "The image API response did not include an image."
            if decode_errors:
                error_message = (
                    "The image API returned invalid base64 payload(s): "
                    + "; ".join(decode_errors)
                )
            return {
                "success": False,
                "error": error_message,
                "model": model,
            }

        return {
            "success": True,
            "prompt": prompt,
            "model": model,
            "requested_model": requested_model,
            "size": size,
            "quality": quality,
            "timeout": timeout,
            "images": images,
            "path": images[0].get("path", ""),
            "url": images[0].get("url", ""),
            "message": f"Generated {len(images)} image(s).",
            "fallback_reason": fallback_reason,
        }

    return [generate_image]


__all__ = ["image_generation_tools"]
