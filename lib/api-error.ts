type ApiErrorDetail = {
  loc?: Array<string | number>;
  msg?: string;
  type?: string;
};

function formatLocation(loc: unknown): string {
  if (!Array.isArray(loc)) return "请求参数";
  const parts = loc
    .filter((part) => part !== "body" && part !== "query" && part !== "path")
    .map(String);
  return parts.length ? parts.join(".") : "请求参数";
}

export function formatApiError(body: unknown, fallback = "请求失败") {
  if (!body || typeof body !== "object") return fallback;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (!item || typeof item !== "object") return String(item);
        const entry = item as ApiErrorDetail;
        return `${formatLocation(entry.loc)}：${entry.msg || entry.type || "参数不合法"}`;
      })
      .filter(Boolean);
    if (messages.length) return messages.join("；");
  }
  if (detail && typeof detail === "object") {
    const entry = detail as ApiErrorDetail;
    return `${formatLocation(entry.loc)}：${entry.msg || entry.type || "参数不合法"}`;
  }
  return fallback;
}
