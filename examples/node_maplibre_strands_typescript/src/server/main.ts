import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Agent, type AgentStreamEvent } from "@strands-agents/sdk";
import express from "express";
import { WebSocket, WebSocketServer } from "ws";
import { BrowserMapSession } from "./mapSession.js";
import { createProviderModel, defaultModelForProvider } from "./models.js";
import {
  createMapLibreTools,
  MAPLIBRE_CODE_PROMPT,
  MAPLIBRE_SYSTEM_PROMPT,
} from "./tools.js";
import { PROVIDER_IDS } from "./types.js";
import type {
  ChatMessage,
  ClientMessage,
  HistoryItem,
  JsonObject,
  ProviderId,
  RuntimeOptions,
} from "./types.js";

const MAX_CONTEXT_MESSAGES = 12;
const MAX_CONTEXT_CHARS = 12_000;

function parseBooleanEnv(value: string | undefined): boolean {
  return value === "1" || value?.toLowerCase() === "true";
}

function parseArgs(argv: string[]): RuntimeOptions {
  const options: RuntimeOptions = {
    host: process.env.HOST || "127.0.0.1",
    port: Number(process.env.PORT || 8765),
    allowBrowserCode: parseBooleanEnv(process.env.ALLOW_BROWSER_CODE),
    allowDestructive: parseBooleanEnv(process.env.ALLOW_DESTRUCTIVE),
    commandTimeoutSeconds: Number(process.env.COMMAND_TIMEOUT_SECONDS || 30),
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--host") {
      options.host = argv[index + 1] || options.host;
      index += 1;
    } else if (arg === "--port") {
      options.port = Number(argv[index + 1] || options.port);
      index += 1;
    } else if (arg === "--allow-browser-code") {
      options.allowBrowserCode = true;
    } else if (arg === "--allow-destructive") {
      options.allowDestructive = true;
    } else if (arg === "--command-timeout") {
      options.commandTimeoutSeconds = Number(
        argv[index + 1] || options.commandTimeoutSeconds,
      );
      index += 1;
    }
  }

  if (!Number.isFinite(options.port) || options.port <= 0) {
    throw new Error("Port must be a positive number.");
  }
  if (
    !Number.isFinite(options.commandTimeoutSeconds) ||
    options.commandTimeoutSeconds <= 0
  ) {
    throw new Error("Command timeout must be a positive number.");
  }
  return options;
}

function providerFromMessage(value: unknown): ProviderId {
  if (
    typeof value === "string" &&
    (PROVIDER_IDS as readonly string[]).includes(value)
  ) {
    return value as ProviderId;
  }
  return "openai-codex";
}

function defaultModelMap(): Record<ProviderId, string> {
  return Object.fromEntries(
    PROVIDER_IDS.map((provider) => [
      provider,
      defaultModelForProvider(provider),
    ]),
  ) as Record<ProviderId, string>;
}

function sendJson(websocket: WebSocket, payload: JsonObject): void {
  if (websocket.readyState !== WebSocket.OPEN) {
    return;
  }
  websocket.send(JSON.stringify(payload));
}

function compactHistory(rawHistory: unknown): HistoryItem[] {
  if (!Array.isArray(rawHistory)) {
    return [];
  }
  return rawHistory
    .filter((item): item is HistoryItem => {
      return Boolean(item && typeof item === "object" && !Array.isArray(item));
    })
    .map((item) => ({
      role: (item.role === "assistant" ? "assistant" : "user") as
        | "assistant"
        | "user",
      text: String(item.text || "").trim(),
      status: item.status,
    }))
    .filter((item) => Boolean(item.text));
}

function buildPromptWithContext(history: HistoryItem[], prompt: string): string {
  const lines: string[] = [];
  for (const item of history.slice(-MAX_CONTEXT_MESSAGES)) {
    const body = String(item.text || "").trim();
    if (!body) {
      continue;
    }
    const role = item.role === "assistant" ? "Assistant" : "User";
    lines.push(`${role}: ${body}`);
  }
  if (!lines.length) {
    return prompt;
  }
  let transcript = lines.join("\n\n");
  if (transcript.length > MAX_CONTEXT_CHARS) {
    transcript = `[Earlier history truncated]\n${transcript.slice(-MAX_CONTEXT_CHARS)}`;
  }
  return [
    "Use the recent conversation history for context. The current user request is the authoritative request to answer now.",
    "",
    "Recent conversation:",
    transcript,
    "",
    `Current user request:\n${prompt}`,
  ].join("\n");
}

function finalTextFromStreamEvent(event: AgentStreamEvent): string | undefined {
  if (event.type !== "agentResultEvent") {
    return undefined;
  }
  return event.result.toString();
}

async function runChatTurn(
  websocket: WebSocket,
  session: BrowserMapSession,
  options: RuntimeOptions,
  message: ChatMessage,
): Promise<void> {
  const provider = providerFromMessage(message.provider);
  const userText = String(message.message || "").trim();
  if (!userText) {
    sendJson(websocket, {
      type: "chat_result",
      ok: false,
      answer: "",
      error: "Chat message is empty.",
    });
    return;
  }

  sendJson(websocket, { type: "chat_status", status: "running" });
  const started = Date.now();
  const executedTools: string[] = [];
  let streamedText = "";
  let finalAnswer = "";

  try {
    const model = await createProviderModel(
      provider,
      String(message.model || defaultModelForProvider(provider)),
    );
    const systemPrompt = options.allowBrowserCode
      ? `${MAPLIBRE_SYSTEM_PROMPT}\n\n${MAPLIBRE_CODE_PROMPT}`
      : MAPLIBRE_SYSTEM_PROMPT;
    const agent = new Agent({
      name: "GeoAgent Node MapLibre",
      model,
      systemPrompt,
      tools: createMapLibreTools(session, {
        allowBrowserCode: options.allowBrowserCode,
        allowDestructive: options.allowDestructive,
      }),
      printer: false,
      toolExecutor: "sequential",
    });
    const prompt = buildPromptWithContext(
      compactHistory(message.history),
      userText,
    );

    for await (const event of agent.stream(prompt)) {
      if (event.type === "modelStreamUpdateEvent") {
        const modelEvent = event.event;
        if (
          modelEvent.type === "modelContentBlockDeltaEvent" &&
          modelEvent.delta.type === "textDelta"
        ) {
          streamedText += modelEvent.delta.text;
          sendJson(websocket, {
            type: "chat_delta",
            text: modelEvent.delta.text,
          });
        }
      } else if (event.type === "beforeToolCallEvent") {
        executedTools.push(event.toolUse.name);
        sendJson(websocket, {
          type: "chat_tool",
          name: event.toolUse.name,
        });
      }
      finalAnswer = finalTextFromStreamEvent(event) || finalAnswer;
    }

    const answer = finalAnswer || streamedText || "Done.";
    sendJson(websocket, {
      type: "chat_result",
      ok: true,
      answer,
      executed_tools: [...new Set(executedTools)],
      execution_time: (Date.now() - started) / 1000,
      provider,
      model: String(message.model || defaultModelForProvider(provider)),
      streamed: true,
    });
  } catch (error) {
    sendJson(websocket, {
      type: "chat_result",
      ok: false,
      answer: "",
      error: error instanceof Error ? error.message : String(error),
      executed_tools: [...new Set(executedTools)],
      execution_time: (Date.now() - started) / 1000,
      provider,
      model: String(message.model || defaultModelForProvider(provider)),
    });
  }
}

function isMapCommandResult(message: ClientMessage): message is JsonObject {
  return message.type === "map_command_result";
}

function isChatMessage(message: ClientMessage): message is ChatMessage {
  return message.type === "chat";
}

function createApp(options: RuntimeOptions): void {
  const app = express();
  const server = createServer(app);
  const wss = new WebSocketServer({ noServer: true });
  const currentFile = fileURLToPath(import.meta.url);
  const publicDir = path.resolve(path.dirname(currentFile), "../../public");

  app.get("/geoagent/health", (_request, response) => {
    response.json({
      ok: true,
      providers: PROVIDER_IDS,
      allowBrowserCode: options.allowBrowserCode,
      allowDestructive: options.allowDestructive,
    });
  });
  app.use(express.static(publicDir));

  server.on("upgrade", (request, socket, head) => {
    if (request.url !== "/geoagent/ws") {
      socket.destroy();
      return;
    }
    wss.handleUpgrade(request, socket, head, (websocket) => {
      wss.emit("connection", websocket, request);
    });
  });

  wss.on("connection", (websocket) => {
    const session = new BrowserMapSession(
      websocket,
      options.commandTimeoutSeconds * 1000,
    );
    let activeChat: Promise<void> | null = null;

    sendJson(websocket, {
      type: "session",
      sessionId: session.sessionId,
      mapId: session.mapId,
      providers: PROVIDER_IDS,
      defaultModels: defaultModelMap(),
      allowBrowserCode: options.allowBrowserCode,
      allowDestructive: options.allowDestructive,
    });

    websocket.on("message", (data) => {
      let raw: unknown;
      try {
        raw = JSON.parse(data.toString());
      } catch {
        sendJson(websocket, {
          type: "error",
          error: "Message was not valid JSON.",
        });
        return;
      }
      if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
        sendJson(websocket, {
          type: "error",
          error: "Message must be a JSON object.",
        });
        return;
      }
      const parsed = raw as ClientMessage;

      if (isMapCommandResult(parsed)) {
        session.resolveResult(parsed);
        return;
      }

      if (!isChatMessage(parsed)) {
        sendJson(websocket, {
          type: "error",
          error: `Unsupported message type: ${String(parsed.type)}`,
        });
        return;
      }

      if (activeChat) {
        sendJson(websocket, {
          type: "chat_result",
          ok: false,
          answer: "",
          error: "A chat turn is already running for this session.",
        });
        return;
      }

      session.mapId = String(parsed.mapId || "default");
      activeChat = runChatTurn(websocket, session, options, parsed).finally(
        () => {
          activeChat = null;
        },
      );
    });

    websocket.on("close", () => {
      session.failAll("Browser WebSocket disconnected.");
    });
  });

  server.listen(options.port, options.host, () => {
    const url = `http://${options.host}:${options.port}`;
    console.log(`GeoAgent Node MapLibre example listening at ${url}`);
    console.log(`WebSocket endpoint: ws://${options.host}:${options.port}/geoagent/ws`);
    console.log(
      `Tools: browserCode=${options.allowBrowserCode ? "on" : "off"}, destructive=${options.allowDestructive ? "on" : "off"}`,
    );
  });
}

createApp(parseArgs(process.argv.slice(2)));
