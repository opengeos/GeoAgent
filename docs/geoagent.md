# GeoAgent

The main `GeoAgent` class is the public facade over the deepagents-based runtime. It builds the agent graph, threads conversation state through a checkpointer, and bridges confirmation-required tool calls through a user-supplied callback.

::: geoagent.core.agent
