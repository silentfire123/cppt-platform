const configuredApiBase = (
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  process.env.NEXT_PUBLIC_MODEL_API_BASE ??
  "/api"
).replace(/\/+$/, "") || "/api";

export function resolveApiBase(configured: string, origin?: string): string {
  if (!origin || !/^https?:\/\//i.test(configured)) return configured;
  try {
    const target = new URL(configured, origin);
    const current = new URL(origin);
    if (target.hostname === current.hostname) {
      return target.pathname.replace(/\/+$/, "") || "/api";
    }
  } catch {
    // Keep the configured value when it is not a valid absolute URL.
  }
  return configured;
}

export const API_BASE = resolveApiBase(
  configuredApiBase,
  typeof window === "undefined" ? undefined : window.location.origin,
);

export function apiUrl(path: string): string {
  return `${API_BASE}/${path.replace(/^\/+/, "")}`;
}

export const AUTH_CALLBACK_URL = (
  process.env.NEXT_PUBLIC_AUTH_CALLBACK_URL ?? apiUrl("/auth/callback")
).replace(/([^:]\/)\/+/g, "$1");

export function resolveApiWebSocketUrl(
  configured: string,
  path: string,
  origin: string,
): string {
  const base = new URL(configured, origin);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/+$/, "")}/`;
  base.search = "";
  base.hash = "";
  return new URL(path.replace(/^\/+/, ""), base).toString();
}

export function apiWebSocketUrl(path: string): string {
  return resolveApiWebSocketUrl(API_BASE, path, window.location.origin);
}

const inferenceBase = (process.env.NEXT_PUBLIC_INFERENCE_BASE_URL ?? "").replace(/\/+$/, "");

export function inferenceUrl(path: string): string {
  return `${inferenceBase}/${path.replace(/^\/+/, "")}`;
}
