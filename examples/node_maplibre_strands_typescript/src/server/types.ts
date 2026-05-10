export type JsonObject = Record<string, unknown>;

export const PROVIDER_IDS = [
  "openai-codex",
  "openai",
  "anthropic",
  "gemini",
  "bedrock",
  "litellm",
  "ollama",
] as const;

export type ProviderId = (typeof PROVIDER_IDS)[number];

export interface HistoryItem {
  role?: "user" | "assistant";
  text?: string;
  status?: "ok" | "error";
}

export interface ChatMessage {
  type: "chat";
  provider?: string;
  model?: string;
  mapId?: string;
  message?: string;
  history?: unknown;
}

export interface MapCommandResultMessage {
  type: "map_command_result";
  id?: string;
  ok?: boolean;
  result?: unknown;
  error?: string;
}

export type ClientMessage = ChatMessage | MapCommandResultMessage | JsonObject;

export interface RuntimeOptions {
  host: string;
  port: number;
  allowBrowserCode: boolean;
  allowDestructive: boolean;
  commandTimeoutSeconds: number;
}
