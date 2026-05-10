import {
  Agent,
  type AgentStreamEvent,
  type JSONValue,
  type Model,
  type Tool,
  tool,
} from "@strands-agents/sdk";
import { AnthropicModel } from "@strands-agents/sdk/models/anthropic";
import { GoogleModel } from "@strands-agents/sdk/models/google";
import { OpenAIModel } from "@strands-agents/sdk/models/openai";
import DOMPurify from "dompurify";
import { marked } from "marked";
import maplibregl, { type ProjectionSpecification, type StyleSpecification } from "maplibre-gl";
import { LayerControl } from "maplibre-gl-layer-control";
import { z } from "zod";
import "./styles.css";
import "maplibre-gl-layer-control/style.css";

type JsonObject = Record<string, unknown>;
type BBox = [number, number, number, number];
interface GeoJsonLayerDefinition {
  id: string;
  suffix: string;
  type: "fill" | "line" | "circle";
  filter: unknown[];
  paint: JsonObject;
}

interface HistoryItem {
  role: "user" | "assistant";
  text: string;
  status?: "ok" | "error";
}

interface Overlay {
  kind: "geojson" | "raster" | "marker";
  name: string;
  sourceIds: string[];
  layerIds: string[];
  marker?: maplibregl.Marker;
  data?: GeoJSON.GeoJSON;
  url?: string;
  style?: JsonObject;
  attribution?: string;
}

type ProviderId = "openai-responses" | "openai-chat" | "anthropic" | "google";

interface ProviderConfig {
  id: ProviderId;
  label: string;
  keyLabel: string;
  keyPlaceholder: string;
  storageKey: string;
  defaultModel: string;
}

const BASEMAPS: Record<string, string | StyleSpecification> = {
  liberty: "https://tiles.openfreemap.org/styles/liberty",
  positron: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  voyager: "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
  demotiles: "https://demotiles.maplibre.org/style.json",
  openstreetmap: "osm",
  osm: {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: [
          "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
          "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
          "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
        ],
        tileSize: 256,
        attribution: "OpenStreetMap contributors",
      },
    },
    layers: [{ id: "osm", type: "raster", source: "osm" }],
  },
};
const DEFAULT_BASEMAP = BASEMAPS.liberty;
const DEFAULT_PROVIDER: ProviderId = "openai-responses";
const PROVIDER_STORAGE_KEY = "geoagent.strands.provider";
const MODEL_STORAGE_PREFIX = "geoagent.strands.model.";
const PROVIDER_CONFIGS: Record<ProviderId, ProviderConfig> = {
  "openai-responses": {
    id: "openai-responses",
    label: "OpenAI Responses",
    keyLabel: "OpenAI API Key",
    keyPlaceholder: "sk-...",
    storageKey: "geoagent.strands.openai_api_key",
    defaultModel: "gpt-5.5",
  },
  "openai-chat": {
    id: "openai-chat",
    label: "OpenAI Chat",
    keyLabel: "OpenAI API Key",
    keyPlaceholder: "sk-...",
    storageKey: "geoagent.strands.openai_api_key",
    defaultModel: "gpt-5.5",
  },
  anthropic: {
    id: "anthropic",
    label: "Anthropic",
    keyLabel: "Anthropic API Key",
    keyPlaceholder: "sk-ant-...",
    storageKey: "geoagent.strands.anthropic_api_key",
    defaultModel: "claude-sonnet-4-6",
  },
  google: {
    id: "google",
    label: "Google Gemini",
    keyLabel: "Gemini API Key",
    keyPlaceholder: "AIza...",
    storageKey: "geoagent.strands.google_api_key",
    defaultModel: "gemini-3.1-pro-preview",
  },
};
const PROVIDER_OPTIONS = Object.values(PROVIDER_CONFIGS)
  .map((provider) => `<option value="${provider.id}">${provider.label}</option>`)
  .join("");
const BROWSER_MAPLIBRE_SYSTEM_PROMPT = `You are an AI assistant embedded in a browser web app with direct access to a live MapLibre map through dedicated browser tools.

Workflow guidance:
- Use browser map tools for map navigation, layer inspection, marker creation, GeoJSON display, layer visibility, feature queries, and screenshots.
- Coordinates in user-facing prompts are latitude/longitude, but browser map internals use longitude/latitude. Use the tool parameter names exactly.
- Do not ask the user to paste JavaScript or run Python for actions that the browser map tools can perform.
- Keep responses concise and include layer names, locations, and tool results when useful.`;
const BROWSER_MAPLIBRE_CODE_SYSTEM_PROMPT = `Browser JavaScript code execution is enabled for this local session. This tool runs arbitrary JavaScript in the page context and is not a safety boundary; treat it as a trusted, local-only escape hatch.

When no dedicated browser map tool can perform the requested MapLibre operation, write a short JavaScript snippet and run it with run_maplibre_script. The snippet executes in the browser with these names in scope: map, maplibregl, and helpers. Prefer MapLibre GL JS API calls, keep code focused on map operations, and avoid credential handling, storage access, unrelated DOM manipulation, or broad network operations.`;

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("Missing #app element.");
}

app.innerHTML = `
  <div id="map" aria-label="MapLibre map"></div>
  <section class="panel" aria-label="GeoAgent chat panel">
    <div class="title">
      <h1>GeoAgent</h1>
      <div class="title-actions">
        <span id="status" class="status connected">Ready</span>
        <button
          id="panel-toggle"
          class="secondary panel-toggle"
          type="button"
          aria-expanded="true"
          aria-label="Collapse panel"
          title="Collapse panel"
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="m18 15-6-6-6 6"></path>
          </svg>
        </button>
      </div>
    </div>

    <div id="panel-body" class="panel-body">
      <div class="settings-grid">
        <label>
          Provider
          <select id="provider-id">${PROVIDER_OPTIONS}</select>
        </label>
        <label>
          <span id="api-key-label">API Key</span>
          <input id="api-key" type="password" autocomplete="off" placeholder="sk-..." />
        </label>
        <label>
          Model
          <input id="model-id" />
        </label>
      </div>

      <div class="toggle-row" aria-label="Agent permissions">
        <label class="checkbox-row">
          <input id="allow-code" type="checkbox" />
          <span>MapLibre JS</span>
        </label>
        <label class="checkbox-row">
          <input id="allow-destructive" type="checkbox" />
          <span>Layer removal</span>
        </label>
      </div>

      <div id="log" class="log" aria-live="polite"></div>

      <form id="chat-form">
        <label>
          Prompt
          <textarea
            id="prompt"
            placeholder="Add a red marker for Knoxville and zoom to it."
          ></textarea>
        </label>
        <div class="actions" style="margin-top: 8px">
          <button id="send" type="submit" disabled>Send</button>
          <button id="clear-log" class="secondary" type="button">Clear</button>
        </div>
      </form>
    </div>
  </section>
`;

const map = new maplibregl.Map({
  container: "map",
  style: DEFAULT_BASEMAP,
  center: [-98.5795, 39.8283],
  zoom: 3,
  maxPitch: 85,
  canvasContextAttributes: { preserveDrawingBuffer: true },
});
map.addControl(new maplibregl.NavigationControl(), "top-right");

let layerControl: LayerControl | null = null;

function basemapStyleUrl(style: string | StyleSpecification): string | undefined {
  return typeof style === "string" && /^https?:\/\//.test(style) ? style : undefined;
}

function removeLayerControl(): void {
  if (layerControl) {
    map.removeControl(layerControl);
    layerControl = null;
  }
}

function installLayerControl(style: string | StyleSpecification): void {
  removeLayerControl();
  const styleUrl = basemapStyleUrl(style);
  layerControl = new LayerControl({
    collapsed: true,
    ...(styleUrl ? { basemapStyleUrl: styleUrl } : {}),
    panelWidth: 320,
    panelMinWidth: 240,
    panelMaxWidth: 420,
  });
  map.addControl(layerControl, "top-right");
}

map.once("load", () => installLayerControl(DEFAULT_BASEMAP));

let geoAgentControlEl: HTMLDivElement | null = null;
class GeoAgentControl {
  onAdd(): HTMLElement {
    const container = document.createElement("div");
    container.className = "maplibregl-ctrl maplibregl-ctrl-group geoagent-map-control";
    const button = document.createElement("button");
    button.type = "button";
    button.title = "Expand GeoAgent";
    button.setAttribute("aria-label", "Expand GeoAgent");
    button.innerHTML = `
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="m6 9 6 6 6-6"></path>
      </svg>
    `;
    button.addEventListener("click", () => setPanelCollapsed(false));
    container.append(button);
    geoAgentControlEl = container;
    return container;
  }

  onRemove(): void {
    geoAgentControlEl?.parentNode?.removeChild(geoAgentControlEl);
    geoAgentControlEl = null;
  }
}
map.addControl(new GeoAgentControl(), "top-left");

const statusEl = requiredElement<HTMLSpanElement>("#status");
const panelEl = requiredElement<HTMLElement>(".panel");
const panelToggleButton = requiredElement<HTMLButtonElement>("#panel-toggle");
const sendButton = requiredElement<HTMLButtonElement>("#send");
const providerSelect = requiredElement<HTMLSelectElement>("#provider-id");
const apiKeyLabel = requiredElement<HTMLSpanElement>("#api-key-label");
const apiKeyInput = requiredElement<HTMLInputElement>("#api-key");
const modelIdInput = requiredElement<HTMLInputElement>("#model-id");
const allowCodeInput = requiredElement<HTMLInputElement>("#allow-code");
const allowDestructiveInput = requiredElement<HTMLInputElement>("#allow-destructive");
const logEl = requiredElement<HTMLDivElement>("#log");
const form = requiredElement<HTMLFormElement>("#chat-form");
const promptEl = requiredElement<HTMLTextAreaElement>("#prompt");
const clearLogButton = requiredElement<HTMLButtonElement>("#clear-log");

let agent: Agent | null = null;
let agentSignature = "";
let busy = false;
let streamingAssistantTextEl: HTMLDivElement | null = null;
let streamingAssistantText = "";
const history: HistoryItem[] = [];
const overlays = new Map<string, Overlay>();

allowCodeInput.checked = false;
const storedProvider = sessionStorage.getItem(PROVIDER_STORAGE_KEY);
if (storedProvider && storedProvider in PROVIDER_CONFIGS) {
  providerSelect.value = storedProvider;
} else {
  providerSelect.value = DEFAULT_PROVIDER;
}
loadProviderSettings();

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}

function numberArg(args: JsonObject, key: string): number {
  const value = args[key];
  if (typeof value !== "number" && typeof value !== "string") {
    throw new Error(`Expected numeric argument: ${key}`);
  }
  return Number(value);
}

function stringArg(args: JsonObject, key: string, fallback = ""): string {
  const value = args[key];
  return typeof value === "string" ? value : fallback;
}

function objectArg(args: JsonObject, key: string): JsonObject {
  const value = args[key];
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

function setStatus(text: string, kind = ""): void {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`.trim();
}

function appendLog(role: string, text: string): HTMLDivElement {
  const entry = document.createElement("div");
  entry.className = "entry";
  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role;
  const textEl = document.createElement("div");
  textEl.className = "text";
  textEl.textContent = text;
  entry.append(roleEl, textEl);
  logEl.append(entry);
  logEl.scrollTop = logEl.scrollHeight;
  return textEl;
}

function renderAssistantMarkdown(element: HTMLElement, markdown: string): void {
  const html = marked(markdown || "", {
    async: false,
    breaks: false,
    gfm: true,
  });
  element.innerHTML = DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
  });
  for (const link of element.querySelectorAll("a")) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
}

function appendAssistantLog(markdown: string): HTMLDivElement {
  const textEl = appendLog("assistant", "");
  textEl.classList.add("markdown");
  renderAssistantMarkdown(textEl, markdown);
  return textEl;
}

function currentProviderId(): ProviderId {
  return providerSelect.value in PROVIDER_CONFIGS
    ? (providerSelect.value as ProviderId)
    : DEFAULT_PROVIDER;
}

function currentProviderConfig(): ProviderConfig {
  return PROVIDER_CONFIGS[currentProviderId()];
}

function modelStorageKey(providerId: ProviderId): string {
  return `${MODEL_STORAGE_PREFIX}${providerId}`;
}

function loadProviderSettings(): void {
  const provider = currentProviderConfig();
  apiKeyLabel.textContent = provider.keyLabel;
  apiKeyInput.placeholder = provider.keyPlaceholder;
  apiKeyInput.value = sessionStorage.getItem(provider.storageKey) || "";
  modelIdInput.value = sessionStorage.getItem(modelStorageKey(provider.id)) || provider.defaultModel;
  modelIdInput.placeholder = provider.defaultModel;
}

function updateControls(): void {
  sendButton.disabled =
    busy || !promptEl.value.trim() || !apiKeyInput.value.trim() || !modelIdInput.value.trim();
}

function setPanelCollapsed(collapsed: boolean): void {
  panelEl.classList.toggle("collapsed", collapsed);
  if (geoAgentControlEl) {
    geoAgentControlEl.style.display = collapsed ? "block" : "none";
  }
  panelToggleButton.setAttribute("aria-expanded", String(!collapsed));
  panelToggleButton.setAttribute(
    "aria-label",
    collapsed ? "Expand panel" : "Collapse panel",
  );
  panelToggleButton.title = collapsed ? "Expand panel" : "Collapse panel";
  const iconPath = panelToggleButton.querySelector("path");
  if (iconPath) {
    iconPath.setAttribute("d", collapsed ? "m6 9 6 6 6-6" : "m18 15-6-6-6 6");
  }
}

function togglePanel(): void {
  setPanelCollapsed(!panelEl.classList.contains("collapsed"));
}

function waitForMapIdle(): Promise<void> {
  return new Promise((resolve) => {
    if (map.loaded()) {
      resolve();
      return;
    }
    map.once("idle", () => resolve());
  });
}

function slug(value: unknown): string {
  return (
    String(value || "layer")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || "layer"
  );
}

function uniqueSourceId(baseId: string): string {
  let sourceId = baseId;
  let index = 2;
  while (map.getSource(sourceId)) {
    sourceId = `${baseId}-${index}`;
    index += 1;
  }
  return sourceId;
}

function uniqueLayerBaseId(baseId: string, suffixes: string[]): string {
  let layerBaseId = baseId;
  let index = 2;
  while (suffixes.some((suffix) => map.getLayer(`${layerBaseId}${suffix}`))) {
    layerBaseId = `${baseId}-${index}`;
    index += 1;
  }
  return layerBaseId;
}

function isOverlayLayerId(layerId: string): boolean {
  return Array.from(overlays.values()).some((overlay) =>
    overlay.layerIds.includes(layerId),
  );
}

function removeOverlay(name: string): boolean {
  const key = Array.from(overlays.keys()).find(
    (item) => item === name || slug(item) === slug(name),
  );
  if (!key) {
    return false;
  }
  const overlay = overlays.get(key);
  if (!overlay) {
    return false;
  }
  for (const layerId of overlay.layerIds) {
    if (map.getLayer(layerId)) {
      map.removeLayer(layerId);
    }
  }
  for (const sourceId of overlay.sourceIds) {
    if (map.getSource(sourceId)) {
      map.removeSource(sourceId);
    }
  }
  overlay.marker?.remove();
  overlays.delete(key);
  return true;
}

function serializableFeature(feature: maplibregl.MapGeoJSONFeature): JsonObject {
  return {
    type: "Feature",
    geometry: feature.geometry ?? null,
    properties: feature.properties ?? {},
    layer: feature.layer
      ? {
          id: feature.layer.id,
          type: feature.layer.type,
          source: feature.layer.source,
        }
      : undefined,
  };
}

function geojsonLayerPaint(style: JsonObject): {
  fill: JsonObject;
  line: JsonObject;
  circle: JsonObject;
} {
  const color =
    stringArg(style, "color") || stringArg(style, "line-color") || "#1c7ed6";
  const fillColor = stringArg(style, "fill-color", stringArg(style, "fillColor", color));
  const lineColor = stringArg(style, "line-color", stringArg(style, "lineColor", color));
  const circleColor = stringArg(
    style,
    "circle-color",
    stringArg(style, "circleColor", color),
  );
  const opacity = Number(style.opacity ?? style["fill-opacity"] ?? 0.35);
  return {
    fill: {
      "fill-color": fillColor,
      "fill-outline-color": lineColor,
      "fill-opacity": Math.max(0, Math.min(1, opacity)),
    },
    line: {
      "line-color": lineColor,
      "line-width": Number(style["line-width"] ?? style.lineWidth ?? 2),
    },
    circle: {
      "circle-color": circleColor,
      "circle-radius": Number(style["circle-radius"] ?? style.radius ?? 6),
      "circle-stroke-color": "#ffffff",
      "circle-stroke-width": 1,
    },
  };
}

async function fetchGeoJson(url: string): Promise<GeoJSON.GeoJSON> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not fetch GeoJSON (${response.status}) from ${url}`);
  }
  return (await response.json()) as GeoJSON.GeoJSON;
}

function extendBounds(bounds: BBox | null, coordinate: unknown): BBox | null {
  if (!Array.isArray(coordinate) || coordinate.length < 2) {
    return bounds;
  }
  const lon = Number(coordinate[0]);
  const lat = Number(coordinate[1]);
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
    return bounds;
  }
  if (!bounds) {
    return [lon, lat, lon, lat];
  }
  bounds[0] = Math.min(bounds[0], lon);
  bounds[1] = Math.min(bounds[1], lat);
  bounds[2] = Math.max(bounds[2], lon);
  bounds[3] = Math.max(bounds[3], lat);
  return bounds;
}

function extendGeometryBounds(
  bounds: BBox | null,
  geometry: GeoJSON.Geometry | null | undefined,
): BBox | null {
  if (!geometry) {
    return bounds;
  }
  if (geometry.type === "GeometryCollection") {
    return geometry.geometries.reduce(
      (currentBounds, item) => extendGeometryBounds(currentBounds, item),
      bounds,
    );
  }
  const visit = (coordinates: unknown): void => {
    if (!Array.isArray(coordinates)) {
      return;
    }
    if (
      coordinates.length >= 2 &&
      typeof coordinates[0] === "number" &&
      typeof coordinates[1] === "number"
    ) {
      bounds = extendBounds(bounds, coordinates);
      return;
    }
    for (const item of coordinates) {
      visit(item);
    }
  };
  visit(geometry.coordinates);
  return bounds;
}

function geoJsonBounds(geojson: GeoJSON.GeoJSON | undefined): BBox | null {
  if (!geojson) {
    return null;
  }
  if (Array.isArray(geojson.bbox) && geojson.bbox.length >= 4) {
    const dimension = geojson.bbox.length / 2;
    const bbox = [
      geojson.bbox[0],
      geojson.bbox[1],
      geojson.bbox[dimension],
      geojson.bbox[dimension + 1],
    ].map(Number);
    if (bbox.every(Number.isFinite)) {
      return bbox as BBox;
    }
  }
  if (geojson.type === "FeatureCollection") {
    return geojson.features.reduce(
      (bounds, feature) => extendGeometryBounds(bounds, feature.geometry),
      null as BBox | null,
    );
  }
  if (geojson.type === "Feature") {
    return extendGeometryBounds(null, geojson.geometry);
  }
  return extendGeometryBounds(null, geojson);
}

function collectGeometryTypes(
  types: Set<GeoJSON.GeoJsonGeometryTypes>,
  geometry: GeoJSON.Geometry | null | undefined,
): Set<GeoJSON.GeoJsonGeometryTypes> {
  if (!geometry) {
    return types;
  }
  if (geometry.type === "GeometryCollection") {
    for (const item of geometry.geometries) {
      collectGeometryTypes(types, item);
    }
    return types;
  }
  types.add(geometry.type);
  return types;
}

function geoJsonGeometryTypes(
  geojson: GeoJSON.GeoJSON | undefined,
): Set<GeoJSON.GeoJsonGeometryTypes> {
  const types = new Set<GeoJSON.GeoJsonGeometryTypes>();
  if (!geojson) {
    return types;
  }
  if (geojson.type === "FeatureCollection") {
    for (const feature of geojson.features) {
      collectGeometryTypes(types, feature.geometry);
    }
    return types;
  }
  if (geojson.type === "Feature") {
    return collectGeometryTypes(types, geojson.geometry);
  }
  return collectGeometryTypes(types, geojson);
}

function geojsonLayerDefs(
  baseId: string,
  paint: ReturnType<typeof geojsonLayerPaint>,
  geojson: GeoJSON.GeoJSON | undefined,
): GeoJsonLayerDefinition[] {
  const geometryTypes = geoJsonGeometryTypes(geojson);
  const hasKnownTypes = geometryTypes.size > 0;
  const hasPolygons =
    !hasKnownTypes ||
    geometryTypes.has("Polygon") ||
    geometryTypes.has("MultiPolygon");
  const hasLines =
    !hasKnownTypes ||
    geometryTypes.has("LineString") ||
    geometryTypes.has("MultiLineString");
  const hasPoints =
    !hasKnownTypes ||
    geometryTypes.has("Point") ||
    geometryTypes.has("MultiPoint");
  const layerDefs: GeoJsonLayerDefinition[] = [];
  if (hasPolygons) {
    layerDefs.push({
      id: `${baseId}-fill`,
      suffix: "-fill",
      type: "fill",
      filter: ["==", ["geometry-type"], "Polygon"],
      paint: paint.fill,
    });
  }
  if (hasLines) {
    layerDefs.push({
      id: `${baseId}-line`,
      suffix: "-line",
      type: "line",
      filter: ["==", ["geometry-type"], "LineString"],
      paint: paint.line,
    });
  }
  if (hasPoints) {
    layerDefs.push({
      id: `${baseId}-point`,
      suffix: "-point",
      type: "circle",
      filter: ["==", ["geometry-type"], "Point"],
      paint: paint.circle,
    });
  }
  return layerDefs;
}

function zoomToGeoJsonBounds(bounds: BBox | null): boolean {
  if (!bounds) {
    return false;
  }
  const [west, south, east, north] = bounds;
  if (![west, south, east, north].every(Number.isFinite)) {
    return false;
  }
  if (west === east && south === north) {
    map.easeTo({
      center: [west, south],
      zoom: Math.max(map.getZoom(), 12),
    });
    return true;
  }
  map.fitBounds(
    [
      [west, south],
      [east, north],
    ],
    { padding: 48, maxZoom: 16 },
  );
  return true;
}

async function addGeoJsonOverlay(overlay: {
  name: string;
  data?: GeoJSON.GeoJSON;
  url?: string;
  style?: JsonObject;
  zoomTo?: boolean;
}): Promise<void> {
  await waitForMapIdle();
  removeOverlay(overlay.name);
  const sourceId = uniqueSourceId(`${slug(overlay.name)}-source`);
  const style = overlay.style ?? {};
  const paint = geojsonLayerPaint(style);
  let sourceData = overlay.data;
  if (!sourceData && overlay.url) {
    try {
      sourceData = await fetchGeoJson(overlay.url);
    } catch (error) {
      console.warn(error);
    }
  }
  const initialLayerDefs = geojsonLayerDefs(slug(overlay.name), paint, sourceData);
  const baseId = uniqueLayerBaseId(
    slug(overlay.name),
    initialLayerDefs.map((item) => item.suffix),
  );
  const layerDefs = geojsonLayerDefs(baseId, paint, sourceData);
  map.addSource(sourceId, {
    type: "geojson",
    data:
      sourceData ??
      overlay.url ??
      ({ type: "FeatureCollection", features: [] } as GeoJSON.FeatureCollection),
  });
  for (const layer of layerDefs) {
    const { suffix: _suffix, ...layerDefinition } = layer;
    map.addLayer({
      ...layerDefinition,
      source: sourceId,
    } as maplibregl.LayerSpecification);
  }
  overlays.set(overlay.name, {
    kind: "geojson",
    name: overlay.name,
    data: overlay.data,
    url: overlay.url,
    style,
    sourceIds: [sourceId],
    layerIds: layerDefs.map((item) => item.id),
  });
  if (overlay.zoomTo) {
    zoomToGeoJsonBounds(geoJsonBounds(sourceData ?? overlay.data));
  }
}

async function addRasterOverlay(overlay: {
  name: string;
  url: string;
  attribution?: string;
}): Promise<void> {
  await waitForMapIdle();
  removeOverlay(overlay.name);
  const sourceId = uniqueSourceId(`${slug(overlay.name)}-source`);
  const layerId = uniqueLayerBaseId(slug(overlay.name), [""]);
  map.addSource(sourceId, {
    type: "raster",
    tiles: [overlay.url],
    tileSize: 256,
    attribution: overlay.attribution ?? "",
  });
  map.addLayer({
    id: layerId,
    type: "raster",
    source: sourceId,
  });
  overlays.set(overlay.name, {
    kind: "raster",
    name: overlay.name,
    url: overlay.url,
    attribution: overlay.attribution,
    sourceIds: [sourceId],
    layerIds: [layerId],
  });
}

function addMarkerOverlay(args: JsonObject): string {
  const name = stringArg(args, "name", `marker-${overlays.size + 1}`);
  removeOverlay(name);
  const marker = new maplibregl.Marker({
    color: stringArg(args, "color", "#3388ff"),
  })
    .setLngLat([numberArg(args, "lon"), numberArg(args, "lat")])
    .addTo(map);
  marker.getElement().title =
    stringArg(args, "tooltip") || stringArg(args, "popup") || name;
  const popup = stringArg(args, "popup");
  if (popup) {
    marker.setPopup(new maplibregl.Popup().setText(popup));
  }
  overlays.set(name, {
    kind: "marker",
    name,
    marker,
    sourceIds: [],
    layerIds: [],
  });
  return name;
}

function serializeScriptResult(value: unknown): unknown {
  if (value === undefined) {
    return null;
  }
  try {
    return JSON.parse(JSON.stringify(value)) as unknown;
  } catch {
    return String(value);
  }
}

async function runMapLibreScript(args: JsonObject): Promise<JsonObject> {
  const code = stringArg(args, "code").trim();
  if (!code) {
    throw new Error("No MapLibre JavaScript code was provided.");
  }
  const description = stringArg(args, "description");
  const overlayNames = (): string[] => Array.from(overlays.keys());
  const removeOverlayGated = (name: string): boolean => {
    requireDestructiveApproval("Layer removal");
    return removeOverlay(name);
  };
  const helpers = Object.freeze({
    overlayNames,
    waitForMapIdle,
    slug,
    removeOverlay: removeOverlayGated,
    addGeoJsonOverlay,
    addRasterOverlay,
    addMarkerOverlay,
    serializeScriptResult,
  });
  const fn = new Function(
    "map",
    "maplibregl",
    "helpers",
    `"use strict"; return (async () => {\n${code}\n})()`,
  ) as (
    map: maplibregl.Map,
    maplibreglApi: typeof maplibregl,
    helpers: Record<string, unknown>,
  ) => Promise<unknown>;
  const result = await fn(map, maplibregl, helpers);
  return {
    success: true,
    message: description || "MapLibre script executed.",
    result: serializeScriptResult(result),
    description,
    maplibre_script: code,
  };
}

async function restoreOverlaysAfterStyleChange(): Promise<void> {
  const saved = Array.from(overlays.values()).map((overlay) => ({ ...overlay }));
  for (const overlay of saved) {
    if (overlay.kind === "geojson") {
      await addGeoJsonOverlay({
        name: overlay.name,
        data: overlay.data,
        url: overlay.url,
        style: overlay.style,
      });
    } else if (overlay.kind === "raster" && overlay.url) {
      await addRasterOverlay({
        name: overlay.name,
        url: overlay.url,
        attribution: overlay.attribution,
      });
    }
  }
}

async function executeCommand(command: string, args: JsonObject = {}): Promise<unknown> {
  await waitForMapIdle();

  if (command === "list_layers") {
    const styleLayers = (map.getStyle().layers ?? []).map((layer) => ({
      id: layer.id,
      type: layer.type,
      source: "source" in layer ? layer.source : null,
      visible: layer.layout?.visibility !== "none",
      user_added: isOverlayLayerId(layer.id),
    }));
    const markerLayers = Array.from(overlays.values())
      .filter((overlay) => overlay.kind === "marker")
      .map((overlay) => ({
        id: overlay.name,
        type: "marker",
        source: null,
        visible: true,
        user_added: true,
      }));
    return [...styleLayers, ...markerLayers];
  }

  if (command === "get_map_state") {
    const center = map.getCenter();
    const bounds = map.getBounds();
    const projection = map.getProjection();
    return {
      center: [center.lng, center.lat],
      zoom: map.getZoom(),
      bearing: map.getBearing(),
      pitch: map.getPitch(),
      projection: projection?.type ?? "mercator",
      bounds: {
        west: bounds.getWest(),
        south: bounds.getSouth(),
        east: bounds.getEast(),
        north: bounds.getNorth(),
      },
      user_layers: Array.from(overlays.keys()),
    };
  }

  if (command === "set_center") {
    map.jumpTo({
      center: [numberArg(args, "lon"), numberArg(args, "lat")],
      zoom: args.zoom == null ? map.getZoom() : Number(args.zoom),
    });
    return "Centered map.";
  }

  if (command === "fly_to") {
    map.flyTo({
      center: [numberArg(args, "lon"), numberArg(args, "lat")],
      zoom: args.zoom == null ? map.getZoom() : Number(args.zoom),
    });
    return "Moved map.";
  }

  if (command === "set_zoom") {
    map.setZoom(numberArg(args, "zoom"));
    return "Zoom updated.";
  }

  if (command === "set_projection") {
    const projection = stringArg(args, "projection", "mercator").trim().toLowerCase();
    if (projection !== "globe" && projection !== "mercator") {
      throw new Error(`Unsupported projection: ${projection}. Use globe or mercator.`);
    }
    map.setProjection({ type: projection } as ProjectionSpecification);
    return `Projection changed to ${projection}.`;
  }

  if (command === "zoom_to_bounds") {
    map.fitBounds(
      [
        [numberArg(args, "west"), numberArg(args, "south")],
        [numberArg(args, "east"), numberArg(args, "north")],
      ],
      { padding: 48, maxZoom: 16 },
    );
    return "Zoomed to bounds.";
  }

  if (command === "change_basemap") {
    const rawStyle = stringArg(args, "style", "liberty").trim();
    let style = BASEMAPS[rawStyle.toLowerCase()] ?? rawStyle;
    if (typeof style === "string" && BASEMAPS[style]) {
      style = BASEMAPS[style];
    }
    removeLayerControl();
    map.setStyle(style);
    await new Promise<void>((resolve) => map.once("style.load", () => resolve()));
    await restoreOverlaysAfterStyleChange();
    installLayerControl(style);
    return `Basemap changed to ${rawStyle}.`;
  }

  if (command === "add_marker") {
    const name = addMarkerOverlay(args);
    return `Added marker ${name}.`;
  }

  if (command === "add_geojson_data") {
    await addGeoJsonOverlay({
      name: stringArg(args, "name", "geojson"),
      data: args.data as GeoJSON.GeoJSON,
      style: objectArg(args, "style"),
      zoomTo: true,
    });
    return `Added GeoJSON layer ${stringArg(args, "name", "geojson")}.`;
  }

  if (command === "add_vector_data") {
    await addGeoJsonOverlay({
      name: stringArg(args, "name", "vector-data"),
      url: stringArg(args, "url"),
      style: objectArg(args, "style"),
      zoomTo: true,
    });
    return `Added GeoJSON URL layer ${stringArg(args, "name", "vector-data")}.`;
  }

  if (command === "add_xyz_tile_layer") {
    await addRasterOverlay({
      name: stringArg(args, "name", "xyz-tiles"),
      url: stringArg(args, "url"),
      attribution: stringArg(args, "attribution"),
    });
    return `Added XYZ tile layer ${stringArg(args, "name", "xyz-tiles")}.`;
  }

  if (command === "set_layer_visibility") {
    const name = stringArg(args, "name");
    const overlay = overlays.get(name);
    const visibility = args.visible ? "visible" : "none";
    if (overlay) {
      for (const layerId of overlay.layerIds) {
        if (map.getLayer(layerId)) {
          map.setLayoutProperty(layerId, "visibility", visibility);
        }
      }
      return `Layer ${name} visibility updated.`;
    }
    if (map.getLayer(name)) {
      map.setLayoutProperty(name, "visibility", visibility);
      return `Layer ${name} visibility updated.`;
    }
    throw new Error(`Layer not found: ${name}`);
  }

  if (command === "set_layer_opacity") {
    const name = stringArg(args, "name");
    const opacity = Math.max(0, Math.min(1, numberArg(args, "opacity")));
    const overlay = overlays.get(name);
    const layerIds = overlay ? overlay.layerIds : [name];
    let changed = false;
    for (const layerId of layerIds) {
      const layer = map.getLayer(layerId);
      if (!layer) {
        continue;
      }
      const prop =
        layer.type === "raster"
          ? "raster-opacity"
          : layer.type === "fill"
            ? "fill-opacity"
            : layer.type === "line"
              ? "line-opacity"
              : layer.type === "circle"
                ? "circle-opacity"
                : null;
      if (prop) {
        map.setPaintProperty(layerId, prop, opacity);
        changed = true;
      }
    }
    if (!changed) {
      throw new Error(`Layer not found or opacity unsupported: ${name}`);
    }
    return `Layer ${name} opacity updated.`;
  }

  if (command === "query_rendered_features") {
    const canvas = map.getCanvas();
    const point: [number, number] =
      args.x == null || args.y == null
        ? [canvas.clientWidth / 2, canvas.clientHeight / 2]
        : [Number(args.x), Number(args.y)];
    const layers: string[] = [];
    if (Array.isArray(args.layers)) {
      for (const requested of args.layers) {
        const layerName = String(requested);
        const overlay = overlays.get(layerName);
        if (overlay) {
          layers.push(...overlay.layerIds);
        } else if (map.getLayer(layerName)) {
          layers.push(layerName);
        }
      }
    }
    return map
      .queryRenderedFeatures(point, layers.length ? { layers } : {})
      .slice(0, 50)
      .map(serializableFeature);
  }

  if (command === "screenshot_map") {
    return {
      data_url: map.getCanvas().toDataURL("image/png"),
      width: map.getCanvas().width,
      height: map.getCanvas().height,
    };
  }

  if (command === "remove_layer") {
    const name = stringArg(args, "name");
    if (!removeOverlay(name)) {
      throw new Error(`User-added layer not found: ${name}`);
    }
    return `Removed layer ${name}.`;
  }

  if (command === "clear_layers") {
    for (const name of Array.from(overlays.keys())) {
      removeOverlay(name);
    }
    return "Cleared user-added layers.";
  }

  if (command === "run_maplibre_script") {
    return runMapLibreScript(args);
  }

  throw new Error(`Unsupported command: ${command}`);
}

function toJsonValue(value: unknown): JSONValue {
  return serializeScriptResult(value) as JSONValue;
}

async function runCommand(command: string, args: unknown = {}): Promise<JSONValue> {
  return toJsonValue(await executeCommand(command, (args ?? {}) as JsonObject));
}

const optionalStyleSchema = z
  .record(z.string(), z.any())
  .optional()
  .describe("Optional MapLibre-compatible paint/style values.");

function requireDestructiveApproval(action: string): void {
  if (!allowDestructiveInput.checked) {
    throw new Error(`${action} is disabled. Enable layer removal first.`);
  }
}

function createMapLibreTools(): Tool[] {
  const tools: Tool[] = [
    tool({
      name: "list_layers",
      description: "List layers currently present in the browser MapLibre map.",
      inputSchema: z.object({}),
      callback: () => runCommand("list_layers"),
    }),
    tool({
      name: "get_map_state",
      description: "Return the browser map camera state, bounds, pitch, bearing, and user layers.",
      inputSchema: z.object({}),
      callback: () => runCommand("get_map_state"),
    }),
    tool({
      name: "set_center",
      description: "Center the browser map on a latitude/longitude coordinate.",
      inputSchema: z.object({
        lat: z.number().describe("Latitude in decimal degrees."),
        lon: z.number().describe("Longitude in decimal degrees."),
        zoom: z.number().optional().describe("Optional zoom level."),
      }),
      callback: (input) => runCommand("set_center", input),
    }),
    tool({
      name: "fly_to",
      description: "Animate the browser map to a latitude/longitude coordinate.",
      inputSchema: z.object({
        lat: z.number().describe("Latitude in decimal degrees."),
        lon: z.number().describe("Longitude in decimal degrees."),
        zoom: z.number().optional().describe("Optional zoom level."),
      }),
      callback: (input) => runCommand("fly_to", input),
    }),
    tool({
      name: "set_zoom",
      description: "Set the browser map zoom level.",
      inputSchema: z.object({
        zoom: z.number().describe("MapLibre zoom level."),
      }),
      callback: (input) => runCommand("set_zoom", input),
    }),
    tool({
      name: "set_projection",
      description: "Switch the browser MapLibre map projection between globe and mercator.",
      inputSchema: z.object({
        projection: z
          .enum(["globe", "mercator"])
          .describe("Projection to use. Use globe for a 3D earth view or mercator for the standard flat map."),
      }),
      callback: (input) => runCommand("set_projection", input),
    }),
    tool({
      name: "zoom_to_bounds",
      description: "Zoom the browser map to a west/south/east/north bounding box.",
      inputSchema: z.object({
        west: z.number(),
        south: z.number(),
        east: z.number(),
        north: z.number(),
      }),
      callback: (input) => runCommand("zoom_to_bounds", input),
    }),
    tool({
      name: "change_basemap",
      description: "Change the browser MapLibre basemap style by URL or known style id.",
      inputSchema: z.object({
        style: z.string().describe("Known style id or MapLibre style URL."),
      }),
      callback: (input) => runCommand("change_basemap", input),
    }),
    tool({
      name: "add_marker",
      description: "Add a marker to the browser map.",
      inputSchema: z.object({
        lat: z.number(),
        lon: z.number(),
        popup: z.string().optional(),
        tooltip: z.string().optional(),
        name: z.string().optional(),
        color: z.string().optional(),
      }),
      callback: (input) => runCommand("add_marker", input),
    }),
    tool({
      name: "add_geojson_data",
      description: "Add an in-memory GeoJSON object to the browser map.",
      inputSchema: z.object({
        data: z.any().describe("GeoJSON Feature, FeatureCollection, or Geometry object."),
        name: z.string().describe("Layer name."),
        style: optionalStyleSchema,
      }),
      callback: (input) => runCommand("add_geojson_data", input),
    }),
    tool({
      name: "add_vector_data",
      description: "Add a GeoJSON URL to the browser map.",
      inputSchema: z.object({
        url: z.string().url().describe("URL to a GeoJSON document."),
        name: z.string().describe("Layer name."),
        style: optionalStyleSchema,
      }),
      callback: (input) => runCommand("add_vector_data", input),
    }),
    tool({
      name: "add_xyz_tile_layer",
      description: "Add an XYZ raster tile layer to the browser map.",
      inputSchema: z.object({
        url: z.string().describe("XYZ tile URL template."),
        name: z.string().describe("Layer name."),
        attribution: z.string().optional(),
      }),
      callback: (input) => runCommand("add_xyz_tile_layer", input),
    }),
    tool({
      name: "set_layer_visibility",
      description: "Show or hide a browser map layer.",
      inputSchema: z.object({
        name: z.string().describe("Layer or overlay name."),
        visible: z.boolean(),
      }),
      callback: (input) => runCommand("set_layer_visibility", input),
    }),
    tool({
      name: "set_layer_opacity",
      description: "Set browser map layer opacity between 0 and 1.",
      inputSchema: z.object({
        name: z.string().describe("Layer or overlay name."),
        opacity: z.number().min(0).max(1),
      }),
      callback: (input) => runCommand("set_layer_opacity", input),
    }),
    tool({
      name: "query_rendered_features",
      description: "Query rendered features from the browser map.",
      inputSchema: z.object({
        layers: z.array(z.string()).optional(),
        x: z.number().optional().describe("Canvas x coordinate; defaults to map center."),
        y: z.number().optional().describe("Canvas y coordinate; defaults to map center."),
      }),
      callback: (input) => runCommand("query_rendered_features", input),
    }),
    tool({
      name: "screenshot_map",
      description: "Capture the browser map canvas as a PNG data URL.",
      inputSchema: z.object({}),
      callback: () => runCommand("screenshot_map"),
    }),
    tool({
      name: "remove_layer",
      description: "Remove a user-added layer from the browser map.",
      inputSchema: z.object({
        name: z.string().describe("User-added layer name."),
      }),
      callback: (input) => {
        requireDestructiveApproval("Layer removal");
        return runCommand("remove_layer", input);
      },
    }),
    tool({
      name: "clear_layers",
      description: "Remove all user-added layers from the browser map.",
      inputSchema: z.object({}),
      callback: () => {
        requireDestructiveApproval("Clearing layers");
        return runCommand("clear_layers");
      },
    }),
  ];

  if (allowCodeInput.checked) {
    tools.push(
      tool({
        name: "run_maplibre_script",
        description:
          "Run a short JavaScript snippet against the live browser MapLibre map when no dedicated tool fits.",
        inputSchema: z.object({
          code: z.string().describe("JavaScript code to execute against map, maplibregl, and helpers."),
          description: z.string().optional(),
        }),
        callback: (input) => runCommand("run_maplibre_script", input),
      }),
    );
  }

  return tools;
}

function invalidateAgent(): void {
  agent = null;
  agentSignature = "";
}

function createProviderModel(providerId: ProviderId, modelId: string, apiKey: string): Model {
  if (providerId === "openai-responses") {
    return new OpenAIModel({
      api: "responses",
      modelId,
      apiKey,
      clientConfig: {
        dangerouslyAllowBrowser: true,
      },
    });
  }
  if (providerId === "openai-chat") {
    return new OpenAIModel({
      api: "chat",
      modelId,
      apiKey,
      clientConfig: {
        dangerouslyAllowBrowser: true,
      },
    });
  }
  if (providerId === "anthropic") {
    return new AnthropicModel({
      modelId,
      apiKey,
      clientConfig: {
        dangerouslyAllowBrowser: true,
      },
    });
  }
  if (providerId === "google") {
    return new GoogleModel({
      modelId,
      apiKey,
    });
  }

  throw new Error(`Unsupported browser provider: ${providerId}`);
}

function getAgent(): Agent {
  const provider = currentProviderConfig();
  const apiKey = apiKeyInput.value.trim();
  const modelId = modelIdInput.value.trim();
  if (!apiKey) {
    throw new Error(`${provider.keyLabel} is required.`);
  }
  if (!modelId) {
    throw new Error("Model id is required.");
  }

  const signature = JSON.stringify({
    providerId: provider.id,
    modelId,
    apiKey,
    allowCode: allowCodeInput.checked,
  });
  if (agent && agentSignature === signature) {
    return agent;
  }

  const systemPrompt = allowCodeInput.checked
    ? `${BROWSER_MAPLIBRE_SYSTEM_PROMPT}\n\n${BROWSER_MAPLIBRE_CODE_SYSTEM_PROMPT}`
    : BROWSER_MAPLIBRE_SYSTEM_PROMPT;
  const model = createProviderModel(provider.id, modelId, apiKey);
  agent = new Agent({
    name: "GeoAgent MapLibre Browser",
    model,
    systemPrompt,
    tools: createMapLibreTools(),
    printer: false,
    toolExecutor: "sequential",
  });
  agentSignature = signature;
  return agent;
}

function appendAssistantDelta(text: string): void {
  if (!text) {
    return;
  }
  if (!streamingAssistantTextEl) {
    streamingAssistantTextEl = appendAssistantLog("");
  }
  streamingAssistantText += text;
  renderAssistantMarkdown(streamingAssistantTextEl, streamingAssistantText);
  logEl.scrollTop = logEl.scrollHeight;
}

function handleAgentStreamEvent(event: AgentStreamEvent): string | undefined {
  if (event.type === "modelStreamUpdateEvent") {
    const modelEvent = event.event;
    if (
      modelEvent.type === "modelContentBlockDeltaEvent" &&
      modelEvent.delta.type === "textDelta"
    ) {
      appendAssistantDelta(modelEvent.delta.text);
    }
    return undefined;
  }

  if (event.type === "beforeToolCallEvent") {
    appendLog("tool", `Running ${event.toolUse.name}`);
    return undefined;
  }

  if (event.type === "agentResultEvent") {
    return event.result.toString();
  }

  return undefined;
}

async function sendPrompt(): Promise<void> {
  const text = promptEl.value.trim();
  if (!text || busy) {
    return;
  }
  history.push({ role: "user", text });
  appendLog("user", text);
  streamingAssistantTextEl = null;
  streamingAssistantText = "";
  promptEl.value = "";
  busy = true;
  setStatus("Running", "connected");
  updateControls();
  try {
    const activeAgent = getAgent();
    let finalAnswer = "";
    for await (const event of activeAgent.stream(text)) {
      finalAnswer = handleAgentStreamEvent(event) ?? finalAnswer;
    }
    const answer = finalAnswer || streamingAssistantText || "Done.";
    const assistantEl = streamingAssistantTextEl as HTMLDivElement | null;
    if (assistantEl) {
      renderAssistantMarkdown(assistantEl, answer);
    } else {
      appendAssistantLog(answer);
    }
    history.push({ role: "assistant", text: answer, status: "ok" });
    setStatus("Ready", "connected");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    appendLog("error", message);
    history.push({ role: "assistant", text: message, status: "error" });
    setStatus("Error", "error");
  } finally {
    busy = false;
    streamingAssistantTextEl = null;
    streamingAssistantText = "";
    updateControls();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void sendPrompt();
});

providerSelect.addEventListener("change", () => {
  sessionStorage.setItem(PROVIDER_STORAGE_KEY, currentProviderId());
  loadProviderSettings();
  invalidateAgent();
  updateControls();
});
apiKeyInput.addEventListener("input", () => {
  const apiKey = apiKeyInput.value.trim();
  const storageKey = currentProviderConfig().storageKey;
  if (apiKey) {
    sessionStorage.setItem(storageKey, apiKey);
  } else {
    sessionStorage.removeItem(storageKey);
  }
  invalidateAgent();
  updateControls();
});
modelIdInput.addEventListener("input", () => {
  const modelId = modelIdInput.value.trim();
  const storageKey = modelStorageKey(currentProviderId());
  if (modelId) {
    sessionStorage.setItem(storageKey, modelId);
  } else {
    sessionStorage.removeItem(storageKey);
  }
  invalidateAgent();
  updateControls();
});
allowCodeInput.addEventListener("change", () => {
  invalidateAgent();
});
promptEl.addEventListener("input", updateControls);
promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    void sendPrompt();
  }
});
panelToggleButton.addEventListener("click", togglePanel);
clearLogButton.addEventListener("click", () => {
  logEl.replaceChildren();
  history.splice(0);
  invalidateAgent();
});

appendLog("system", "Browser-only Strands MapLibre agent ready.");
updateControls();
