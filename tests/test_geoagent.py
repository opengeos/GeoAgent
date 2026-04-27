#!/usr/bin/env python

"""Public-API smoke tests for the v1 GeoAgent package."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch


class TestImports(unittest.TestCase):
    """Confirm the documented public API is importable and well-shaped."""

    def test_package_import(self) -> None:
        import geoagent

        self.assertIsNotNone(geoagent.__version__)

    def test_v1_facade_exports(self) -> None:
        from geoagent import (
            GeoAgent,
            GeoAgentContext,
            GeoAgentResponse,
            create_geo_agent,
            for_anymap,
            for_leafmap,
            for_qgis,
        )

        self.assertTrue(hasattr(GeoAgent, "chat"))
        self.assertTrue(hasattr(GeoAgent, "search"))
        self.assertTrue(hasattr(GeoAgent, "analyze"))
        self.assertTrue(hasattr(GeoAgent, "visualize"))
        self.assertTrue(callable(create_geo_agent))
        self.assertTrue(callable(for_leafmap))
        self.assertTrue(callable(for_anymap))
        self.assertTrue(callable(for_qgis))
        # Dataclasses construct without args.
        self.assertIsNotNone(GeoAgentContext())
        self.assertIsNotNone(GeoAgentResponse())

    def test_v1_decorators_and_safety(self) -> None:
        from geoagent import (
            ConfirmCallback,
            ConfirmRequest,
            auto_approve_all,
            auto_approve_safe_only,
            build_interrupt_on,
            geo_tool,
            get_geo_meta,
            needs_confirmation,
            stamp_geo_meta,
        )

        # Smoke: ConfirmRequest is a dataclass we can construct directly.
        req = ConfirmRequest(tool_name="t", args={}, description="", category=None)
        self.assertFalse(auto_approve_safe_only(req))
        self.assertTrue(auto_approve_all(req))
        self.assertEqual(build_interrupt_on([]), {})
        self.assertTrue(callable(geo_tool))
        self.assertTrue(callable(get_geo_meta))
        self.assertTrue(callable(needs_confirmation))
        self.assertTrue(callable(stamp_geo_meta))
        # ConfirmCallback is an alias; just confirm it imports.
        self.assertIsNotNone(ConfirmCallback)

    def test_llm_module(self) -> None:
        from geoagent.core.llm import PROVIDERS, get_default_llm, get_llm

        for provider in ("openai", "anthropic", "google", "ollama"):
            self.assertIn(provider, PROVIDERS)
        self.assertTrue(callable(get_llm))
        self.assertTrue(callable(get_default_llm))

    def test_catalog_registry(self) -> None:
        from geoagent.catalogs.registry import CatalogRegistry

        reg = CatalogRegistry()
        catalogs = reg.list_catalogs()
        self.assertGreater(len(catalogs), 0)
        names = [c.name for c in catalogs]
        self.assertIn("earth_search", names)
        self.assertIn("planetary_computer", names)

    def test_subagent_factories(self) -> None:
        from geoagent.agents import (
            PLANNER_SUBAGENT,
            analysis_subagent,
            context_subagent,
            data_subagent,
            default_subagents,
        )
        from geoagent.core.context import GeoAgentContext

        for sub in (
            PLANNER_SUBAGENT,
            data_subagent(),
            analysis_subagent(),
            context_subagent(),
        ):
            for key in ("name", "description", "system_prompt", "tools"):
                self.assertIn(key, sub)

        names = {s["name"] for s in default_subagents(GeoAgentContext())}
        self.assertEqual(
            {"planner", "data", "analysis", "context"} - names,
            set(),
            f"Expected the four core subagents in default_subagents; got {names}",
        )

    def test_check_api_keys(self) -> None:
        from geoagent.core.llm import check_api_keys

        keys = check_api_keys()
        self.assertIsInstance(keys, dict)
        self.assertIn("openai", keys)

    def test_add_custom_catalog(self) -> None:
        from geoagent.catalogs.registry import CatalogRegistry

        reg = CatalogRegistry()
        reg.add_catalog(
            name="test_catalog",
            url="https://example.com/stac",
            description="Test catalog",
        )
        cat = reg.get_catalog("test_catalog")
        self.assertIsNotNone(cat)
        self.assertEqual(cat.url, "https://example.com/stac")

    def test_get_default_llm_picks_first_cloud_provider(self) -> None:
        """First cloud provider whose env var is set should win."""
        from geoagent.core import llm as llm_mod

        sentinel = object()
        captured: list[str] = []

        def fake_get_llm(provider, **kwargs):
            captured.append(provider)
            return sentinel

        with patch.dict(os.environ, {}, clear=False):
            for key in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
                os.environ.pop(key, None)
            os.environ["OPENAI_API_KEY"] = "test-key"
            with patch.object(llm_mod, "get_llm", side_effect=fake_get_llm):
                result = llm_mod.get_default_llm()

        self.assertIs(result, sentinel)
        self.assertEqual(captured, ["openai"])

    def test_get_default_llm_skips_provider_with_missing_package(self) -> None:
        """ImportError on a cloud provider should fall through and surface in the warning."""
        from geoagent.core import llm as llm_mod

        ollama_sentinel = object()

        def fake_get_llm(provider, **kwargs):
            if provider == "openai":
                raise ImportError("fake langchain-openai missing")
            if provider == "ollama":
                return ollama_sentinel
            raise AssertionError(f"unexpected provider {provider!r}")

        with patch.dict(os.environ, {}, clear=False):
            for key in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
                os.environ.pop(key, None)
            os.environ["OPENAI_API_KEY"] = "test-key"
            with patch.object(llm_mod, "get_llm", side_effect=fake_get_llm):
                with self.assertLogs(llm_mod.logger.name, level="WARNING") as cap:
                    result = llm_mod.get_default_llm()

        self.assertIs(result, ollama_sentinel)
        joined = "\n".join(cap.output)
        # Per-provider skip log + the fallback warning should both mention the
        # missing package, not just "no API key found".
        self.assertIn("openai", joined)
        self.assertIn("missing", joined.lower())

    def test_get_default_llm_raises_when_nothing_usable(self) -> None:
        """No cloud key set and Ollama also unavailable -> RuntimeError."""
        from geoagent.core import llm as llm_mod

        def fake_get_llm(provider, **kwargs):
            raise ImportError(f"{provider} package missing")

        with patch.dict(os.environ, {}, clear=False):
            for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
                os.environ.pop(key, None)
            with patch.object(llm_mod, "get_llm", side_effect=fake_get_llm):
                with self.assertRaises(RuntimeError):
                    llm_mod.get_default_llm()

    def test_resolve_model_routes_provider_colon_model_to_init_chat_model(
        self,
    ) -> None:
        """`model='provider:name'` with no `provider` arg should hit init_chat_model."""
        import langchain.chat_models as chat_models_mod
        from geoagent.core import llm as llm_mod

        sentinel = object()
        with patch.object(
            chat_models_mod, "init_chat_model", return_value=sentinel
        ) as mock_init:
            result = llm_mod.resolve_model(model="anthropic:claude-sonnet-4-5")

        self.assertIs(result, sentinel)
        mock_init.assert_called_once_with("anthropic:claude-sonnet-4-5")


if __name__ == "__main__":
    unittest.main()
