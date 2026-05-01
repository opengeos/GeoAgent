# Response

`GeoAgentResponse` is the dataclass returned by `GeoAgent.chat`. It includes
`answer_text`, `success`, `error_message`, `execution_time`, tool execution
metadata, raw provider result access, and multimodal response artifacts. Image
content blocks returned by supported models are exposed through `images`, while
the original top-level assistant content blocks are available as
`content_blocks`.

::: geoagent.core.result
