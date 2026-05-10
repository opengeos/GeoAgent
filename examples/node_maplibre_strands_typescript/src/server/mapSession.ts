import { randomUUID } from "node:crypto";
import { WebSocket } from "ws";
import type { JsonObject } from "./types.js";

interface PendingCommand {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
}

export class BrowserMapSession {
  readonly sessionId = randomUUID();
  mapId = "default";

  private readonly pending = new Map<string, PendingCommand>();

  constructor(
    private readonly websocket: WebSocket,
    private readonly timeoutMilliseconds: number,
  ) {}

  call(command: string, args: JsonObject = {}): Promise<unknown> {
    if (this.websocket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("Browser WebSocket is not connected."));
    }
    const id = randomUUID();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(
          new Error(
            `Browser did not return a result for ${JSON.stringify(command)} within ${this.timeoutMilliseconds / 1000} seconds.`,
          ),
        );
      }, this.timeoutMilliseconds);
      this.pending.set(id, { resolve, reject, timer });
      this.websocket.send(
        JSON.stringify({
          type: "map_command",
          id,
          sessionId: this.sessionId,
          mapId: this.mapId,
          command,
          args,
        }),
        (error) => {
          if (!error) {
            return;
          }
          clearTimeout(timer);
          this.pending.delete(id);
          reject(error);
        },
      );
    });
  }

  resolveResult(message: JsonObject): boolean {
    const id = String(message.id || "");
    if (!id) {
      return false;
    }
    const pending = this.pending.get(id);
    if (!pending) {
      return false;
    }
    clearTimeout(pending.timer);
    this.pending.delete(id);
    if (message.ok === true) {
      pending.resolve(message.result);
    } else {
      pending.reject(
        new Error(
          String(message.error || `Browser command result ${id} failed.`),
        ),
      );
    }
    return true;
  }

  failAll(error: string): void {
    for (const [id, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(new Error(error));
      this.pending.delete(id);
    }
  }
}
