import { tool, type JSONValue, type Tool } from "@strands-agents/sdk";
import { z } from "zod";
import type { BrowserMapSession } from "./mapSession.js";
import type { JsonObject } from "./types.js";

export const MAPLIBRE_SYSTEM_PROMPT = `You are an AI assistant embedded in a browser web app with access to a live MapLibre map through browser tools.

Workflow guidance:
- Use browser map tools for map navigation, layer inspection, marker creation, GeoJSON display, layer visibility, feature queries, and screenshots.
- Coordinates in user-facing prompts are latitude/longitude, but browser map internals use longitude/latitude. Use the tool parameter names exactly.
- Do not ask the user to paste JavaScript or run Python for actions that the browser map tools can perform.
- If a requested operation has no browser map tool, explain the limitation briefly rather than trying to execute arbitrary JavaScript.
- Keep responses concise and include layer names, locations, and tool results when useful.`;

export const MAPLIBRE_CODE_PROMPT = `Browser JavaScript code execution is enabled for this local session.

When no dedicated browser map tool can perform the requested MapLibre operation, write a short JavaScript snippet and run it with run_maplibre_script. The snippet executes in the browser with these names in scope: map, maplibregl, and helpers. Prefer MapLibre GL JS API calls, keep code focused on map operations, and avoid credential handling, storage access, unrelated DOM manipulation, or broad network operations.`;

const optionalStyleSchema = z
  .record(z.string(), z.any())
  .optional()
  .describe("Optional MapLibre-compatible paint/style values.");

function toJsonValue(value: unknown): JSONValue {
  if (value === undefined) {
    return null;
  }
  try {
    return JSON.parse(JSON.stringify(value)) as JSONValue;
  } catch {
    return String(value);
  }
}

function callMap(
  session: BrowserMapSession,
  command: string,
  args?: unknown,
): Promise<JSONValue> {
  return session
    .call(command, (args ?? {}) as JsonObject)
    .then((value) => toJsonValue(value));
}

export function createMapLibreTools(
  session: BrowserMapSession,
  options: {
    allowBrowserCode: boolean;
    allowDestructive: boolean;
  },
): Tool[] {
  const tools: Tool[] = [
    tool({
      name: "list_layers",
      description: "List layers currently present in the browser MapLibre map.",
      inputSchema: z.object({}),
      callback: () => callMap(session, "list_layers"),
    }),
    tool({
      name: "get_map_state",
      description:
        "Return the browser map camera state, bounds, pitch, bearing, projection, and user layers.",
      inputSchema: z.object({}),
      callback: () => callMap(session, "get_map_state"),
    }),
    tool({
      name: "set_center",
      description: "Center the browser map on a latitude/longitude coordinate.",
      inputSchema: z.object({
        lat: z.number().describe("Latitude in decimal degrees."),
        lon: z.number().describe("Longitude in decimal degrees."),
        zoom: z.number().optional().describe("Optional zoom level."),
      }),
      callback: (input) => callMap(session, "set_center", input),
    }),
    tool({
      name: "fly_to",
      description:
        "Animate the browser map to a latitude/longitude coordinate.",
      inputSchema: z.object({
        lat: z.number().describe("Latitude in decimal degrees."),
        lon: z.number().describe("Longitude in decimal degrees."),
        zoom: z.number().optional().describe("Optional zoom level."),
      }),
      callback: (input) => callMap(session, "fly_to", input),
    }),
    tool({
      name: "set_zoom",
      description: "Set the browser map zoom level.",
      inputSchema: z.object({
        zoom: z.number().describe("MapLibre zoom level."),
      }),
      callback: (input) => callMap(session, "set_zoom", input),
    }),
    tool({
      name: "set_projection",
      description:
        "Switch the browser MapLibre map projection between globe and mercator.",
      inputSchema: z.object({
        projection: z
          .enum(["globe", "mercator"])
          .describe(
            "Projection to use. Use globe for a 3D earth view or mercator for the standard flat map.",
          ),
      }),
      callback: (input) => callMap(session, "set_projection", input),
    }),
    tool({
      name: "zoom_to_bounds",
      description:
        "Zoom the browser map to a west/south/east/north bounding box.",
      inputSchema: z.object({
        west: z.number(),
        south: z.number(),
        east: z.number(),
        north: z.number(),
      }),
      callback: (input) => callMap(session, "zoom_to_bounds", input),
    }),
    tool({
      name: "change_basemap",
      description:
        "Change the browser MapLibre basemap style by URL or known style id.",
      inputSchema: z.object({
        style: z.string().describe("Known style id or MapLibre style URL."),
      }),
      callback: (input) => callMap(session, "change_basemap", input),
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
      callback: (input) => callMap(session, "add_marker", input),
    }),
    tool({
      name: "add_geojson_data",
      description: "Add an in-memory GeoJSON object to the browser map.",
      inputSchema: z.object({
        data: z
          .any()
          .describe("GeoJSON Feature, FeatureCollection, or Geometry object."),
        name: z.string().describe("Layer name."),
        style: optionalStyleSchema,
      }),
      callback: (input) => callMap(session, "add_geojson_data", input),
    }),
    tool({
      name: "add_vector_data",
      description: "Add a GeoJSON URL to the browser map.",
      inputSchema: z.object({
        url: z.string().url().describe("URL to a GeoJSON document."),
        name: z.string().describe("Layer name."),
        style: optionalStyleSchema,
      }),
      callback: (input) => callMap(session, "add_vector_data", input),
    }),
    tool({
      name: "add_xyz_tile_layer",
      description: "Add an XYZ raster tile layer to the browser map.",
      inputSchema: z.object({
        url: z.string().describe("XYZ tile URL template."),
        name: z.string().describe("Layer name."),
        attribution: z.string().optional(),
      }),
      callback: (input) => callMap(session, "add_xyz_tile_layer", input),
    }),
    tool({
      name: "set_layer_visibility",
      description: "Show or hide a browser map layer.",
      inputSchema: z.object({
        name: z.string().describe("Layer or overlay name."),
        visible: z.boolean(),
      }),
      callback: (input) => callMap(session, "set_layer_visibility", input),
    }),
    tool({
      name: "set_layer_opacity",
      description: "Set browser map layer opacity between 0 and 1.",
      inputSchema: z.object({
        name: z.string().describe("Layer or overlay name."),
        opacity: z.number().min(0).max(1),
      }),
      callback: (input) => callMap(session, "set_layer_opacity", input),
    }),
    tool({
      name: "query_rendered_features",
      description: "Query rendered features from the browser map.",
      inputSchema: z.object({
        layers: z.array(z.string()).optional(),
        x: z
          .number()
          .optional()
          .describe("Canvas x coordinate; defaults to map center."),
        y: z
          .number()
          .optional()
          .describe("Canvas y coordinate; defaults to map center."),
      }),
      callback: (input) => callMap(session, "query_rendered_features", input),
    }),
    tool({
      name: "screenshot_map",
      description: "Capture the browser map canvas as a PNG data URL.",
      inputSchema: z.object({}),
      callback: () => callMap(session, "screenshot_map"),
    }),
  ];

  if (options.allowDestructive) {
    tools.push(
      tool({
        name: "remove_layer",
        description: "Remove a user-added layer from the browser map.",
        inputSchema: z.object({
          name: z.string().describe("User-added layer name."),
        }),
        callback: (input) => callMap(session, "remove_layer", input),
      }),
      tool({
        name: "clear_layers",
        description: "Remove all user-added layers from the browser map.",
        inputSchema: z.object({}),
        callback: () => callMap(session, "clear_layers"),
      }),
    );
  }

  if (options.allowBrowserCode) {
    tools.push(
      tool({
        name: "run_maplibre_script",
        description:
          "Run a short JavaScript snippet against the live browser MapLibre map when no dedicated tool fits.",
        inputSchema: z.object({
          code: z
            .string()
            .describe(
              "JavaScript code to execute against map, maplibregl, and helpers.",
            ),
          description: z.string().optional(),
        }),
        callback: (input) => callMap(session, "run_maplibre_script", input),
      }),
    );
  }

  return tools;
}

