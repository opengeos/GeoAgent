import { constants as fsConstants } from "node:fs";
import { access, mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { homedir, platform } from "node:os";
import path from "node:path";

const OPENAI_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token";
const OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const OPENAI_CODEX_SCOPE = "openid profile email offline_access";
const OPENAI_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex";
const REFRESH_SKEW_SECONDS = 300;

interface TokenPayload {
  access_token?: string;
  refresh_token?: string;
  expires_at?: number | string;
  expires_in?: number | string;
  token_type?: string;
  account_id?: string;
  id_token?: string;
  [key: string]: unknown;
}

export interface CodexCredentials {
  accessToken: string;
  baseUrl: string;
  accountId?: string;
}

function defaultTokenFile(): string {
  const override = process.env.GEOAGENT_CODEX_TOKEN_FILE?.trim();
  if (override) {
    return path.resolve(expandHome(override));
  }
  if (platform() === "win32") {
    const root =
      process.env.APPDATA || path.join(homedir(), "AppData", "Roaming");
    return path.join(root, "geoagent", "openai_codex_oauth.json");
  }
  const root = process.env.XDG_CONFIG_HOME || path.join(homedir(), ".config");
  return path.join(root, "geoagent", "openai_codex_oauth.json");
}

function expandHome(value: string): string {
  return value === "~" || value.startsWith("~/")
    ? path.join(homedir(), value.slice(2))
    : value;
}

function tokenExpiresSoon(expiresAt: unknown): boolean {
  if (!expiresAt) {
    return true;
  }
  const expiry = Number(expiresAt);
  if (!Number.isFinite(expiry)) {
    return true;
  }
  return expiry <= Date.now() / 1000 + REFRESH_SKEW_SECONDS;
}

function decodeJwtPayload(token: string): Record<string, unknown> {
  const parts = token.split(".");
  if (parts.length < 2) {
    return {};
  }
  try {
    const payload = parts[1] ?? "";
    const padded = payload + "=".repeat((4 - (payload.length % 4)) % 4);
    const decoded = Buffer.from(padded, "base64url").toString("utf8");
    const parsed = JSON.parse(decoded) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

function extractAccountId(payload: TokenPayload): string {
  for (const key of ["access_token", "id_token"] as const) {
    const claims = decodeJwtPayload(String(payload[key] || ""));
    const direct = claims["https://api.openai.com/auth.chatgpt_account_id"];
    if (typeof direct === "string" && direct.trim()) {
      return direct.trim();
    }
    const auth = claims["https://api.openai.com/auth"];
    if (auth && typeof auth === "object" && !Array.isArray(auth)) {
      const nested = (auth as Record<string, unknown>).chatgpt_account_id;
      if (typeof nested === "string" && nested.trim()) {
        return nested.trim();
      }
    }
    for (const fallback of ["chatgpt_account_id", "account_id"]) {
      const value = claims[fallback];
      if (typeof value === "string" && value.trim()) {
        return value.trim();
      }
    }
  }
  return "";
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function loadStoredToken(tokenFile: string): Promise<TokenPayload> {
  if (!(await pathExists(tokenFile))) {
    throw new Error(
      "OpenAI Codex is not logged in. Run `npm run codex:login`, `geoagent codex login`, or set OPENAI_CODEX_ACCESS_TOKEN.",
    );
  }
  const raw = await readFile(tokenFile, "utf8");
  const parsed = JSON.parse(raw) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`Stored Codex token file is invalid: ${tokenFile}`);
  }
  return parsed as TokenPayload;
}

async function saveStoredToken(
  tokenFile: string,
  payload: TokenPayload,
): Promise<void> {
  await mkdir(path.dirname(tokenFile), { recursive: true });
  const tmpPath = `${tokenFile}.${process.pid}.tmp`;
  try {
    await writeFile(tmpPath, `${JSON.stringify(payload, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600,
    });
    await rename(tmpPath, tokenFile);
  } catch (error) {
    await unlink(tmpPath).catch(() => undefined);
    throw error;
  }
}

async function postForm(
  url: string,
  body: Record<string, string>,
): Promise<TokenPayload> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(body),
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`OAuth token request failed: ${response.status} ${text}`);
  }
  const parsed = JSON.parse(text) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("OAuth token endpoint did not return JSON.");
  }
  return normalizeTokenResponse(parsed as TokenPayload);
}

function normalizeTokenResponse(payload: TokenPayload): TokenPayload {
  const accessToken = String(payload.access_token || "").trim();
  if (!accessToken) {
    throw new Error("OAuth token endpoint did not return an access token.");
  }
  const expiresIn = Number(payload.expires_in ?? 3600);
  const normalized: TokenPayload = {
    ...payload,
    expires_at:
      Math.floor(Date.now() / 1000) +
      Math.max(Number.isFinite(expiresIn) ? expiresIn : 3600, 0),
    token_type: String(payload.token_type || "Bearer"),
  };
  const accountId = extractAccountId(normalized);
  if (accountId) {
    normalized.account_id = accountId;
  }
  return normalized;
}

async function refreshToken(payload: TokenPayload): Promise<TokenPayload> {
  const refreshTokenValue = String(payload.refresh_token || "").trim();
  if (!refreshTokenValue) {
    throw new Error(
      "OpenAI Codex token expired and no refresh token is stored. Run `npm run codex:login` again.",
    );
  }
  const refreshed = await postForm(OPENAI_CODEX_TOKEN_URL, {
    grant_type: "refresh_token",
    client_id: OPENAI_CODEX_CLIENT_ID,
    refresh_token: refreshTokenValue,
    scope: OPENAI_CODEX_SCOPE,
  });
  if (!refreshed.refresh_token) {
    refreshed.refresh_token = refreshTokenValue;
  }
  return refreshed;
}

export async function loadCodexCredentials(): Promise<CodexCredentials> {
  const envToken = process.env.OPENAI_CODEX_ACCESS_TOKEN?.trim();
  const baseUrl =
    process.env.OPENAI_CODEX_BASE_URL?.trim() || OPENAI_CODEX_BASE_URL;
  if (envToken) {
    const accountId = process.env.OPENAI_CODEX_ACCOUNT_ID?.trim();
    return {
      accessToken: envToken,
      baseUrl,
      ...(accountId ? { accountId } : {}),
    };
  }

  const tokenFile = defaultTokenFile();
  let payload = await loadStoredToken(tokenFile);
  if (tokenExpiresSoon(payload.expires_at)) {
    payload = await refreshToken(payload);
    await saveStoredToken(tokenFile, payload);
  }
  const accessToken = String(payload.access_token || "").trim();
  if (!accessToken) {
    throw new Error("Stored OpenAI Codex token payload has no access token.");
  }
  const accountId =
    String(payload.account_id || "").trim() || extractAccountId(payload);
  return {
    accessToken,
    baseUrl,
    ...(accountId ? { accountId } : {}),
  };
}

