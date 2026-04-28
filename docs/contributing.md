# Contributing

Contributions are welcome. GeoAgent is intended to be a shared integration
layer for the geospatial Python and QGIS ecosystem, so high-quality package
adapters, tests, and documentation are especially valuable.

## Ways to Contribute

- Report bugs and reproducible failures.
- Improve existing adapters for `leafmap`, `anymap`, QGIS, STAC, Earthdata,
  Earth Engine, `geoai`, or related packages.
- Add integrations for new geospatial packages and tools.
- Improve mock objects so integrations can be tested in CI.
- Improve docs, examples, notebooks, and prompt patterns.
- Review pull requests and help keep APIs consistent.

Use GitHub issues for bug reports and feature proposals:
<https://github.com/opengeos/GeoAgent/issues>

## Development Setup

GeoAgent uses `pyproject.toml` and supports Python 3.11 and later.

```bash
git clone https://github.com/opengeos/GeoAgent.git
cd GeoAgent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Install optional extras when working on a specific integration:

```bash
pip install -e ".[leafmap,openai,dev]"
pip install -e ".[anymap,anthropic,dev]"
pip install -e ".[stac,earthdata,dev]"
```

For QGIS work, use the Python environment bundled with QGIS or make sure your
development environment can import the relevant QGIS modules. Unit tests should
still pass without QGIS installed by using mocks and lazy imports.

## Local Checks

Run the standard checks before opening a pull request:

```bash
ruff check geoagent tests
pytest -q
```

If you change documentation, build the docs locally when possible:

```bash
mkdocs build
```

Keep changes focused. Avoid unrelated formatting churn or broad refactors in
pull requests that add one adapter or fix one behavior.

## Pull Request Guidelines

Before submitting a pull request:

- Add or update tests for the behavior you changed.
- Update README and docs when the user-facing tool surface changes.
- Keep optional integrations import-safe when their dependencies are missing.
- Use `@geo_tool` for tools exposed to the agent.
- Mark destructive, persistent, costly, or long-running tools with
  confirmation metadata.
- Preserve existing factory behavior unless the pull request is explicitly
  about changing it.
- Include a concise summary of what changed and how you tested it.

## Adding a New Package or Tool Integration

GeoAgent integrations should feel consistent no matter which geospatial package
they wrap. Use this checklist when adding a package such as `geemap`, `xarray`,
`rasterio`, `pyogrio`, `lonboard`, `duckdb`, `whitebox`, a QGIS plugin, or a
domain-specific service client.

### 1. Decide the Integration Shape

Use a **tool module** when the package exposes reusable functions that do not
need a long-lived object:

```text
geoagent/tools/my_package.py
```

Use a **factory** when tools need a live object, session, widget, client, or
QGIS interface:

```python
def my_package_tools(obj: Any) -> list[Any]:
    ...

def for_my_package(obj: Any, ...) -> GeoAgent:
    ...
```

Factories should bind the live object through closures. Do not expose widgets,
clients, credentials, database handles, or QGIS objects as tool arguments.

### 2. Keep Optional Dependencies Optional

Do not import heavy optional packages at module import time unless they are
required by GeoAgent core. Prefer lazy imports inside tool bodies:

```python
@geo_tool(category="raster")
def inspect_raster(path: str) -> dict:
    """Inspect a raster dataset."""
    import rasterio

    with rasterio.open(path) as src:
        return {"width": src.width, "height": src.height}
```

This keeps `import geoagent` and `import geoagent.tools.<module>` usable in CI
and in environments that do not have every geospatial stack installed.

If the integration needs a new dependency, add it under
`[project.optional-dependencies]` in `pyproject.toml`, not as a required core
dependency.

### 3. Design Tool Functions for LLM Use

Good tools are small, explicit, and typed:

- Use clear function names: `add_cog_layer`, `search_stac`, `clip_raster`.
- Use typed parameters and simple return values: `dict`, `list`, `str`,
  numbers, booleans.
- Prefer structured data over long prose.
- Keep docstrings accurate; Strands uses them as tool descriptions.
- Use domain defaults that are safe and unsurprising.
- Validate ambiguous user input and return actionable messages.
- Avoid requiring the model to pass live Python objects.

Example:

```python
from typing import Any

from geoagent.core.decorators import geo_tool


def my_map_tools(m: Any) -> list[Any]:
    """Build tools bound to a live map-like object."""

    @geo_tool(category="map")
    def add_geojson(path_or_url: str, name: str) -> str:
        """Add a GeoJSON dataset to the active map."""
        m.add_geojson(path_or_url, layer_name=name)
        return f"Added GeoJSON layer {name!r}."

    return [add_geojson]
```

### 4. Use Metadata for Safety and Filtering

Use `@geo_tool` metadata deliberately:

```python
@geo_tool(
    category="processing",
    requires_confirmation=True,
    long_running=True,
)
def run_expensive_job(parameters: dict[str, Any]) -> dict[str, Any]:
    """Run an expensive external processing job."""
    ...
```

Mark tools as confirmation-required when they:

- delete or overwrite data;
- save projects or files;
- trigger long-running processing;
- call paid APIs or services;
- mutate remote state;
- make changes that are hard to undo.

Use `available_in=("full", "fast")` only for tools that are safe, quick, and
useful in low-latency mode.

### 5. Register the Integration

For a tool-only integration, export the tool factory from
`geoagent/tools/__init__.py` if it is part of the public API.

For a new top-level factory:

1. Add the adapter module under `geoagent/tools/`.
2. Add a `for_<package>` factory in `geoagent/core/factory.py`.
3. Add it to `geoagent/__init__.py` and `__all__`.
4. Update `assemble_tools` if the integration should participate in shared
   context assembly.
5. Add optional dependencies in `pyproject.toml`.
6. Add docs and examples.

Keep new factories similar to `for_leafmap`, `for_anymap`, and `for_qgis`:

- accept `config`, `model`, `provider`, `model_id`, `fast`, `confirm`, and
  `extra_tools` where appropriate;
- create a `GeoAgentContext`;
- assemble and register tools;
- return a `GeoAgent`.

### 6. Add Mocks and Tests

Every integration should be testable without requiring a full desktop GIS,
cloud account, or heavyweight service.

Recommended tests:

- module imports without optional dependencies installed;
- factory returns an empty list or graceful result for `None` where applicable;
- expected tool names are registered;
- safety metadata is correct;
- tools call the expected mock methods;
- ambiguous and missing layer/resource names are handled;
- destructive tools require confirmation;
- fast mode includes only intended tools;
- docs examples do not reference missing tool names.

Place lightweight mocks in `geoagent/testing/_mocks.py` when they are useful
across tests. Keep mocks intentionally small: they should verify GeoAgent calls
the right methods and handles state correctly, not fully emulate external
packages.

### 7. Document the Integration

Update the relevant docs:

- `README.md` for major public features.
- `docs/index.md` for package-level overview changes.
- `docs/tools.md` for new tool surfaces.
- API docs pages if new modules or factories should appear in navigation.
- Example notebooks or snippets for common workflows.

Document:

- installation extras;
- required environment variables;
- factory usage;
- tool names and their purpose;
- safety/confirmation behavior;
- any limitations or package-version assumptions.

## Integration Quality Bar

An integration is ready when:

- importing GeoAgent still works without the optional package installed;
- the tool API is small, typed, and documented;
- live objects are captured by closures, not exposed as tool arguments;
- user-visible mutations have confirmation where appropriate;
- tests pass with mocks in CI;
- docs show installation, setup, and a minimal working example;
- behavior is consistent with existing `leafmap`, `anymap`, and QGIS adapters.

## Reporting Bugs

Please include:

- operating system and Python version;
- GeoAgent version or commit SHA;
- package extras installed;
- relevant provider and model;
- QGIS version if applicable;
- minimal code to reproduce the issue;
- full traceback or failing command output.

## Proposing Features

For new integrations, open an issue first when the scope is large. Include:

- package or service to integrate;
- main user workflows;
- proposed tool names;
- required optional dependencies;
- whether tools mutate state, call paid services, or run long jobs;
- testing strategy without requiring real credentials or desktop applications.

Small improvements and focused bug fixes can go directly to a pull request.
