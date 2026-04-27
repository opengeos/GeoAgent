#!/usr/bin/env python

"""Public-API smoke tests for the v1 GeoAgent package."""

from __future__ import annotations

import unittest


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


if __name__ == "__main__":
    unittest.main()
