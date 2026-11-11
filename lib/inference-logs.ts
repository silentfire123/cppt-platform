export type InferenceLogItem = {
  created_at?: string | null;
  createdAt?: string | null;
  level?: string | null;
  source?: string | null;
  message?: string | null;
};

export type InferenceLogsPayload = {
  logs?: unknown;
  items?: unknown;
};

type LogSeverity = "WARN" | "ERROR" | null;

function nonEmpty(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function formatLogTimestamp(value: string): string {
  const match = value.match(/(?:T|\s)(\d{2}:\d{2}:\d{2})/);
  if (match) return match[1];
  const timestamp = new Date(value);
  if (!Number.isNaN(timestamp.getTime())) return timestamp.toTimeString().slice(0, 8);
  return value;
}

function stripTerminalNoise(value: string): string {
  return value
    .replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "")
    .replace(/\[\d+(?:;\d+)*m/g, "")
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, "")
    .replace(/\r/g, "")
    .trim();
}

function cleanRuntimeMessage(value: string): { message: string; severity: LogSeverity } {
  let message = stripTerminalNoise(value);
  let severity: LogSeverity = null;

  if (/\b(?:ERROR|CRITICAL|FATAL)\b/i.test(message)) severity = "ERROR";
  else if (/\bWARN(?:ING)?\b/i.test(message)) severity = "WARN";

  // Remote vLLM prefixes are useful to operators but duplicate the platform
  // timestamp and add process/file details that are not useful in the UI.
  // 前缀可能叠加（"[remote] (APIServer pid=701)"），循环剥离直到稳定。
  for (let pass = 0; pass < 3; pass += 1) {
    const before = message;
    message = message
      .replace(/^\s*\[(?:backend|trainer|vllm|remote)\]\s*/i, "")
      .replace(/^\s*\((?:[^)]*?\s+)?pid=\d+\)\s*/i, "")
      .replace(/^\s*\[[^\]]*?pid=\d+[^\]]*\]\s*/i, "")
      .replace(/^\s*(?:\[(?:backend|trainer|vllm|remote|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\]\s*)+/i, "")
      .replace(/^\s*(?:DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\s+\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[[^\]]+\]\s*/i, "")
      .replace(/^\s*(?:DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL):\s*/i, "")
      .replace(/^\s*(?:DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\s+/i, "")
      .replace(/\s+/g, " ")
      .trim();
    if (message === before) break;
  }

  return { message, severity };
}

function formatDisplayLine(timestamp: string | null, message: string, level?: string | null): string | null {
  const cleaned = cleanRuntimeMessage(message);
  if (!cleaned.message) return null;
  const explicitLevel = (level ?? "").toUpperCase();
  const severity: LogSeverity = explicitLevel === "ERROR" ? "ERROR" : explicitLevel === "WARN" || explicitLevel === "WARNING" ? "WARN" : cleaned.severity;
  const levelLabel = severity === "ERROR" ? "错误：" : severity === "WARN" ? "警告：" : "";
  const prefix = timestamp ? `[${formatLogTimestamp(timestamp)}] ` : "";
  return `${prefix}${levelLabel}${cleaned.message}`;
}

function normalizeLegacyLine(line: string): string | null {
  const clean = stripTerminalNoise(line);
  const timestamp = clean.match(/^\[(\d{2}:\d{2}:\d{2})\]\s*/)?.[1] ?? null;
  const message = timestamp ? clean.slice(timestamp.length + 2).trim() : clean;
  return formatDisplayLine(timestamp, message);
}

export function normalizeInferenceLogs(payload: InferenceLogsPayload): string[] {
  if (Array.isArray(payload.logs)) {
    const legacyLogs = payload.logs
      .filter((line): line is string => typeof line === "string")
      .map(normalizeLegacyLine)
      .filter((line): line is string => Boolean(line));
    if (legacyLogs.length) return legacyLogs;
  }
  if (!Array.isArray(payload.items)) return [];
  return payload.items.flatMap((rawItem) => {
    if (!rawItem || typeof rawItem !== "object") return [];
    const item = rawItem as InferenceLogItem;
    const message = nonEmpty(item.message);
    if (!message) return [];
    const timestamp = nonEmpty(item.created_at) ?? nonEmpty(item.createdAt);
    const level = nonEmpty(item.level);
    const source = nonEmpty(item.source);
    const line = formatDisplayLine(timestamp, message, level);
    return line ? [line] : [];
  });
}
