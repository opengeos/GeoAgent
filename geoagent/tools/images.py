"""Image generation tools for GeoAgent."""

from __future__ import annotations

import base64
import binascii
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from geoagent.core.decorators import geo_tool

DEFAULT_IMAGE_MODEL = "gpt-image-2"
FALLBACK_IMAGE_MODEL = "gpt-image-1"
DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_IMAGE_QUALITY = "low"
SUPPORTED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}
SUPPORTED_IMAGE_QUALITIES = {"low", "medium", "high", "auto"}


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


def image_generation_tools() -> list[Any]:
    """Return tools for generating standalone image files."""

    @geo_tool(
        category="image_generation",
        description=(
            "Generate an image from a text prompt using the OpenAI Images API. "
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
        model: str = DEFAULT_IMAGE_MODEL,
        output_dir: str = "",
    ) -> dict[str, Any]:
        """Generate an image file from a text prompt.

        Args:
            prompt: Visual description of the image to generate.
            size: One of 1024x1024, 1024x1536, 1536x1024, or auto.
            quality: One of low, medium, high, or auto.
            model: OpenAI image model to use. Defaults to gpt-image-2.
            output_dir: Optional directory for the generated image.

        Returns:
            A JSON-friendly result containing local image file paths and image
            metadata. The host UI can render the returned ``images`` list.
        """
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"success": False, "error": "Image prompt is empty."}
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
        requested_model = (
            str(model or DEFAULT_IMAGE_MODEL).strip() or DEFAULT_IMAGE_MODEL
        )
        model = requested_model

        from openai import OpenAI

        client = OpenAI()
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
                    "error": f"Image generation failed with {model}: {exc}",
                    "model": model,
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
            "images": images,
            "path": images[0].get("path", ""),
            "url": images[0].get("url", ""),
            "message": f"Generated {len(images)} image(s).",
            "fallback_reason": fallback_reason,
        }

    return [generate_image]


__all__ = ["image_generation_tools"]
