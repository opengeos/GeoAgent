# Response

`GeoAgentResponse` is the dataclass returned by `GeoAgent.chat`. It preserves the v0.x field names (`plan`, `data`, `analysis`, `map`, `code`, `answer_text`, `success`, `error_message`, `execution_time`) and adds `executed_tools`, `cancelled_tools`, and the raw `messages` list for advanced inspection.

::: geoagent.core.result
