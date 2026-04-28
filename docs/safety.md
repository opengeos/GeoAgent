# Safety and confirmation

GeoAgent marks tools with `requires_confirmation`, `destructive`, or `long_running` metadata. At runtime, `ConfirmationHookProvider` handles Strands `BeforeToolCallEvent` and calls your `ConfirmCallback`; denied calls set `cancel_tool` on the event.

::: geoagent.core.safety
