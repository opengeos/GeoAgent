# Subagents

Phase 2 replaced the v0.x 4-agent class hierarchy with declarative deepagents subagent specs. Each module here exposes either a constant `*_SUBAGENT` dict (no runtime state) or a factory function returning one. `default_subagents(ctx)` assembles the active list from the runtime `GeoAgentContext`.

## Coordinator

::: geoagent.agents.coordinator

## Planner

::: geoagent.agents.planner

## Data

::: geoagent.agents.data

## Analysis

::: geoagent.agents.analysis

## Context

::: geoagent.agents.context

## Mapping

::: geoagent.agents.mapping

## QGIS

::: geoagent.agents.qgis

## GeoAI

::: geoagent.agents.geoai

## NASA Earthdata

::: geoagent.agents.earthdata
