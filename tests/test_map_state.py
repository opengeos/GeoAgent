"""Unit tests for MapLibre view_state normalization."""

from __future__ import annotations

from geoagent.tools._map_state import deep_plain, map_state_from_widget


def test_deep_plain_sw_ne() -> None:
    d = {
        "center": {"lng": -1.0, "lat": 2.0},
        "bounds": {
            "_sw": {"lng": -3.0, "lat": 1.0},
            "_ne": {"lng": -2.0, "lat": 2.0},
        },
    }
    out = deep_plain(d)
    assert out["bounds"]["_sw"]["lng"] == -3.0


def test_map_state_from_view_state() -> None:
    class M:
        layers = []
        _style = "x"
        view_state = {"zoom": 4, "center": {"lng": 0, "lat": 1}}

    s = map_state_from_widget(M())
    assert s["zoom"] == 4
    assert s["view_state"]["zoom"] == 4
