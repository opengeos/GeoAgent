import { BedrockModel } from "@strands-agents/sdk/models/bedrock";
import { AnthropicModel } from "@strands-agents/sdk/models/anthropic";
import { GoogleModel } from "@strands-agents/sdk/models/google";
import { OpenAIModel } from "@strands-agents/sdk/models/openai";
import type { Model } from "@strands-agents/sdk";
import { loadCodexCredentials } from "./codexAuth.js";
import type { ProviderId } from "./types.js";

const DEFAULT_CODEX_MODEL = "gpt-5.5";
const DEFAULT_OPENAI_MODEL = "gpt-5.5";
const DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6";
const DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview";
const DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6";
const DEFAULT_LITELLM_MODEL = "openai/gpt-5.5";
const DEFAULT_OLLAMA_MODEL = "qwen3.5:4b";

export function defaultModelForProvider(provider: ProviderId): string {
  switch (provider) {
    case "openai-codex":
      return process.env.OPENAI_CODEX_MODEL || DEFAULT_CODEX_MODEL;
    case "openai":
      return process.env.OPENAI_MODEL || DEFAULT_OPENAI_MODEL;
    case "anthropic":
      return process.env.ANTHROPIC_MODEL || DEFAULT_ANTHROPIC_MODEL;
    case "gemini":
      return (
        process.env.GEMINI_MODEL ||
        process.env.GOOGLE_MODEL ||
        DEFAULT_GEMINI_MODEL
      );
    case "bedrock":
      return process.env.BEDROCK_MODEL || DEFAULT_BEDROCK_MODEL;
    case "litellm":
      return process.env.LITELLM_MODEL || DEFAULT_LITELLM_MODEL;
    case "ollama":
      return process.env.OLLAMA_MODEL || DEFAULT_OLLAMA_MODEL;
  }
}

function requireEnv(name: string, provider: ProviderId): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`${provider} provider requires ${name}.`);
  }
  return value;
}

function openAiCompatibleBaseUrl(
  rawUrl: string,
  options: { appendV1: boolean },
): string {
  const trimmed = rawUrl.replace(/\/+$/, "");
  if (!options.appendV1 || trimmed.endsWith("/v1")) {
    return trimmed;
  }
  return `${trimmed}/v1`;
}

export async function createProviderModel(
  provider: ProviderId,
  modelId: string,
): Promise<Model> {
  const resolvedModel = modelId.trim() || defaultModelForProvider(provider);

  if (provider === "openai-codex") {
    const credentials = await loadCodexCredentials();
    const defaultHeaders: Record<string, string> = {
      "User-Agent": "codex-cli",
    };
    if (credentials.accountId) {
      defaultHeaders["ChatGPT-Account-Id"] = credentials.accountId;
    }
    return new OpenAIModel({
      api: "responses",
      modelId: resolvedModel,
      apiKey: credentials.accessToken,
      clientConfig: {
        baseURL: credentials.baseUrl,
        defaultHeaders,
      },
    });
  }

  if (provider === "openai") {
    return new OpenAIModel({
      api: "responses",
      modelId: resolvedModel,
      apiKey: requireEnv("OPENAI_API_KEY", provider),
    });
  }

  if (provider === "anthropic") {
    return new AnthropicModel({
      modelId: resolvedModel,
      apiKey: requireEnv("ANTHROPIC_API_KEY", provider),
      temperature: 0,
    });
  }

  if (provider === "gemini") {
    return new GoogleModel({
      modelId: resolvedModel,
      apiKey:
        process.env.GEMINI_API_KEY?.trim() ||
        requireEnv("GOOGLE_API_KEY", provider),
      params: { temperature: 0 },
    });
  }

  if (provider === "bedrock") {
    const region =
      process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION || undefined;
    return new BedrockModel({
      modelId: resolvedModel,
      ...(region ? { region } : {}),
      temperature: 0,
    });
  }

  if (provider === "litellm") {
    const baseURL = openAiCompatibleBaseUrl(
      process.env.LITELLM_BASE_URL || "http://127.0.0.1:4000/v1",
      { appendV1: false },
    );
    return new OpenAIModel({
      api: "chat",
      modelId: resolvedModel,
      apiKey: process.env.LITELLM_API_KEY?.trim() || "litellm",
      clientConfig: { baseURL },
    });
  }

  const baseURL = openAiCompatibleBaseUrl(
    process.env.OLLAMA_HOST || "http://127.0.0.1:11434",
    { appendV1: true },
  );
  return new OpenAIModel({
    api: "chat",
    modelId: resolvedModel,
    apiKey: process.env.OLLAMA_API_KEY?.trim() || "ollama",
    clientConfig: { baseURL },
  });
}
