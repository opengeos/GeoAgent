"""Tests for the HSAE GeoAgent tool adapter.

All tests run without ``hydrosovereign`` installed and without any network
access — tools fall back to a deterministic seeded heuristic when the package
is absent, so CI always passes.

Test coverage:
  - Module import safety (no hydrosovereign, no QGIS)
  - Factory returns the expected 9 tools
  - Tool names match the expected surface
  - Safety / confirmation metadata is correct
  - fast-mode availability is correct
  - Basin name resolution (aliases, case, Arabic)
  - Each of the 6 indices returns the correct schema and numeric range
  - UNWC compliance screen returns structured findings
  - Negotiation tool requires confirmation
  - Ambiguous / unknown basin names are handled gracefully
  - _legal_status threshold boundaries are correct
"""

from __future__ import annotations

import importlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Import safety — module must load without hydrosovereign or QGIS
# ---------------------------------------------------------------------------


def test_hsae_module_imports_without_hydrosovereign() -> None:
    """Verify the HSAE adapter is import-safe when hydrosovereign is absent."""
    importlib.import_module("geoagent.tools.hsae")

    assert "geoagent.tools.hsae" in sys.modules


def test_hsae_module_import_does_not_trigger_hydrosovereign() -> None:
    """Importing geoagent.tools.hsae must not eagerly import hydrosovereign."""
    # Remove from sys.modules so we get a clean import
    sys.modules.pop("geoagent.tools.hsae", None)
    # Also remove hydrosovereign if present so we can detect a fresh import
    was_present = sys.modules.pop("hydrosovereign", None)
    try:
        importlib.import_module("geoagent.tools.hsae")
        assert (
            "hydrosovereign" not in sys.modules
        ), "hydrosovereign was eagerly imported by geoagent.tools.hsae"
    finally:
        # Restore hydrosovereign if it was previously imported
        if was_present is not None:
            sys.modules["hydrosovereign"] = was_present


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

from geoagent.tools.hsae import (  # noqa: E402
    hsae_tools,
    _resolve_basin_id,
    _legal_status,
)


def test_hsae_tools_returns_list() -> None:
    tools = hsae_tools()
    assert isinstance(tools, list)


def test_hsae_tools_returns_nine_tools() -> None:
    tools = hsae_tools()
    assert len(tools) == 9


def test_hsae_tools_expose_expected_surface() -> None:
    """Verify all 9 expected tool names are present."""
    tools = {t.tool_name: t for t in hsae_tools()}
    expected = {
        "analyze_basin_compliance",
        "compute_atdi",
        "compute_afsf",
        "compute_ahifd",
        "compute_atci",
        "compute_conflict_index",
        "compute_adts",
        "run_unwc_compliance",
        "get_negotiation_recommendation",
    }
    assert expected == set(tools.keys())


# ---------------------------------------------------------------------------
# Safety metadata
# ---------------------------------------------------------------------------


def test_negotiation_tool_requires_confirmation() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    meta = tools["get_negotiation_recommendation"]._geoagent_meta
    assert meta.requires_confirmation is True


def test_analyze_basin_compliance_is_long_running() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    meta = tools["analyze_basin_compliance"]._geoagent_meta
    assert meta.long_running is True


def test_run_unwc_compliance_is_long_running() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    meta = tools["run_unwc_compliance"]._geoagent_meta
    assert meta.long_running is True


def test_fast_mode_tools_are_subset() -> None:
    """compute_atdi, compute_afsf, compute_ahifd, compute_atci, compute_adts
    should be available in fast mode for quick dashboard queries."""
    tools = {t.tool_name: t for t in hsae_tools()}
    fast_expected = {
        "compute_atdi",
        "compute_afsf",
        "compute_ahifd",
        "compute_atci",
        "compute_adts",
    }
    for name in fast_expected:
        meta = tools[name]._geoagent_meta
        assert "fast" in meta.available_in, f"{name} should be available_in 'fast' mode"


def test_heavy_tools_not_in_fast_mode() -> None:
    """Heavy / confirmation-required tools must not appear in fast mode."""
    tools = {t.tool_name: t for t in hsae_tools()}
    not_fast = {"analyze_basin_compliance", "get_negotiation_recommendation"}
    for name in not_fast:
        meta = tools[name]._geoagent_meta
        assert "fast" not in meta.available_in, f"{name} should NOT be in fast mode"


def test_all_tools_have_hydrology_category() -> None:
    for tool in hsae_tools():
        meta = tool._geoagent_meta
        assert (
            meta.category == "hydrology"
        ), f"{tool.tool_name} has category {meta.category!r}, expected 'hydrology'"


# ---------------------------------------------------------------------------
# Basin name resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias,expected_id",
    [
        ("Blue Nile", "blue_nile_gerd"),
        ("blue nile gerd", "blue_nile_gerd"),
        ("GERD", "blue_nile_gerd"),
        ("النيل الأزرق", "blue_nile_gerd"),
        ("Mekong", "mekong_lancang"),
        ("euphrates", "euphrates_ataturk"),
        ("Danube", "danube_gabcikovo"),
        ("Rhine", "rhine_ijssel"),
    ],
)
def test_basin_alias_resolution(alias: str, expected_id: str) -> None:
    assert _resolve_basin_id(alias) == expected_id


def test_basin_unknown_name_returns_slug() -> None:
    """Unknown basin names should return a slug, not raise."""
    result = _resolve_basin_id("Hypothetical River X")
    assert isinstance(result, str)
    assert len(result) > 0


def test_basin_resolution_is_case_insensitive() -> None:
    assert _resolve_basin_id("MEKONG") == _resolve_basin_id("mekong")
    assert _resolve_basin_id("Blue Nile") == _resolve_basin_id("blue nile")


# ---------------------------------------------------------------------------
# Legal status thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "atdi,expected_status_fragment",
    [
        (10.0, "Compliant"),
        (25.0, "Equitable Use Risk"),
        (40.0, "Significant Harm"),
        (55.0, "Critical"),
        (43.5, "Significant Harm"),  # Blue Nile validated value
    ],
)
def test_legal_status_thresholds(atdi: float, expected_status_fragment: str) -> None:
    status = _legal_status(atdi)
    assert (
        expected_status_fragment in status["status"]
    ), f"ATDI={atdi} → {status['status']!r}, expected fragment {expected_status_fragment!r}"


def test_legal_status_returns_triggered_articles() -> None:
    status = _legal_status(43.5)
    assert "Art." in status["triggered_articles"]


def test_legal_status_compliant_no_articles() -> None:
    status = _legal_status(5.0)
    assert status["triggered_articles"] == "—"


# ---------------------------------------------------------------------------
# Individual tool schemas
# ---------------------------------------------------------------------------


def test_compute_atdi_schema() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    result = tools["compute_atdi"]("Blue Nile")
    assert "ATDI_pct" in result
    assert "legal_status" in result
    assert "triggered_articles" in result
    assert 0.0 <= result["ATDI_pct"] <= 100.0


def test_compute_afsf_schema() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    result = tools["compute_afsf"]("Mekong")
    assert "AFSF_pct" in result
    assert "rolling_window_days" in result
    assert result["rolling_window_days"] == 30
    assert 0.0 <= result["AFSF_pct"] <= 100.0


def test_compute_ahifd_schema() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    result = tools["compute_ahifd"]("Euphrates")
    assert "AHIFD_pct" in result
    assert "art5_triggered" in result
    assert "art7_triggered" in result
    assert isinstance(result["art5_triggered"], bool)
    assert 0.0 <= result["AHIFD_pct"] <= 100.0


def test_compute_atci_schema() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    result = tools["compute_atci"]("Danube")
    assert "ATCI_pct" in result
    assert "compliance_level" in result
    assert result["compliance_level"] in ("Strong", "Partial", "Weak")
    assert 0.0 <= result["ATCI_pct"] <= 100.0


def test_compute_conflict_index_schema() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    result = tools["compute_conflict_index"]("Blue Nile")
    assert "CI_score" in result
    assert "level" in result
    assert result["level"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL")
    assert "art33_recommended" in result
    assert isinstance(result["art33_recommended"], bool)


def test_compute_adts_schema() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    result = tools["compute_adts"]("Blue Nile")
    assert "ADTS_pct" in result
    assert "ATDI_pct" in result
    # ADTS + ATDI must sum to 100
    assert abs(result["ADTS_pct"] + result["ATDI_pct"] - 100.0) < 0.01


def test_adts_atdi_complement_identity() -> None:
    """ADTS = 100 − ATDI must hold exactly for the same basin."""
    tools = {t.tool_name: t for t in hsae_tools()}
    adts_result = tools["compute_adts"]("Mekong")
    atdi_result = tools["compute_atdi"]("Mekong")
    assert abs(adts_result["ADTS_pct"] - (100.0 - atdi_result["ATDI_pct"])) < 0.05


# ---------------------------------------------------------------------------
# Composite tools
# ---------------------------------------------------------------------------


def test_run_unwc_compliance_schema() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    result = tools["run_unwc_compliance"]("Blue Nile")
    assert "overall_compliance" in result
    assert result["overall_compliance"] in ("Compliant", "Partial", "Non-compliant")
    assert "findings" in result
    assert isinstance(result["findings"], list)
    assert "indices_summary" in result
    assert "ATDI" in result["indices_summary"]
    assert "convention" in result


def test_analyze_basin_compliance_full_schema() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    result = tools["analyze_basin_compliance"]("Blue Nile")
    assert "indices" in result
    indices = result["indices"]
    for key in (
        "ATDI_pct",
        "AFSF_pct",
        "AHIFD_pct",
        "ATCI_pct",
        "CI_score",
        "ADTS_pct",
    ):
        assert key in indices, f"Missing index {key}"
    assert 0.0 <= indices["ATDI_pct"] <= 100.0


def test_get_negotiation_recommendation_schema() -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    result = tools["get_negotiation_recommendation"]("Blue Nile")
    assert "p_negotiation_pct" in result
    assert "recommended_pathway" in result
    assert result["recommended_pathway"] in ("Art.17", "Art.33")
    assert "disclaimer" in result
    assert 0.0 <= result["p_negotiation_pct"] <= 100.0


# ---------------------------------------------------------------------------
# Numeric range guards (all indices must be in [0, 100])
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basin", ["Blue Nile", "Mekong", "Euphrates", "Rhine"])
def test_all_indices_in_valid_range(basin: str) -> None:
    tools = {t.tool_name: t for t in hsae_tools()}
    result = tools["analyze_basin_compliance"](basin)
    indices = result["indices"]
    for k, v in indices.items():
        assert 0.0 <= float(v) <= 100.0, f"{k} = {v} out of [0, 100] for basin {basin}"


# ---------------------------------------------------------------------------
# __init__.py registration smoke test
# ---------------------------------------------------------------------------


def test_hsae_tools_exportable_from_tools_package() -> None:
    """Verify hsae_tools is importable from geoagent.tools after registration."""
    from geoagent.tools import hsae_tools as _ht

    assert callable(_ht)
