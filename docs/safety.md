# Safety and Confirmation

GeoAgent classifies tools as either safe (read-only inspection, navigation, preview) or confirmation-required (remove, save, export, download, run long jobs). Confirmation-required tools are wired into deepagents' `interrupt_on` so the agent pauses before they execute. The host application supplies a `ConfirmCallback` to bridge those interrupts to its UI.

::: geoagent.core.safety
