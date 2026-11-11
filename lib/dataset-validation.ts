export type DatasetStructure = "SFT" | "DPO";

export type DatasetInspection = {
  errors: string[];
  warnings: string[];
  rows: number;
};

export function inspectTextDataset(
  text: string,
  structure: DatasetStructure,
): DatasetInspection {
  const errors: string[] = [];
  const warnings: string[] = [];
  const lines = text.split(/\r?\n/);
  let rows = 0;

  lines.forEach((line, index) => {
    if (!line.trim()) return;
    rows += 1;
    const lineNumber = index + 1;
    let row: unknown;
    try {
      row = JSON.parse(line);
    } catch {
      errors.push(`第 ${lineNumber} 行不是合法 JSON。`);
      return;
    }
    if (!row || typeof row !== "object" || Array.isArray(row)) {
      errors.push(`第 ${lineNumber} 行必须是 JSON 对象。`);
      return;
    }
    const record = row as Record<string, unknown>;
    if (structure === "DPO") {
      if ("prompt" in record) {
        for (const field of ["prompt", "chosen", "rejected"]) {
          if (typeof record[field] !== "string" || !record[field].trim()) {
            errors.push(`第 ${lineNumber} 行 ${field} 字段不能为空。`);
          }
        }
        return;
      }
      if (!Array.isArray(record.messages) || !record.messages.length) {
        errors.push(`第 ${lineNumber} 行 messages 必须是非空消息数组。`);
        return;
      }
      if (
        !(record.messages as Array<Record<string, unknown>>).some(
          (message) => message?.role === "user",
        )
      ) {
        errors.push(`第 ${lineNumber} 行 messages 至少需要一条 user 消息。`);
      }
      (record.messages as Array<Record<string, unknown>>).forEach(
        (message, messageIndex) => {
          if (!message || typeof message !== "object") {
            errors.push(
              `第 ${lineNumber} 行 messages 第 ${messageIndex + 1} 条消息必须是对象。`,
            );
            return;
          }
          if (!["system", "user", "assistant"].includes(String(message.role))) {
            errors.push(
              `第 ${lineNumber} 行 messages 第 ${messageIndex + 1} 条消息 role 不支持。`,
            );
          }
          if (
            typeof message.content !== "string" ||
            !message.content.trim()
          ) {
            errors.push(
              `第 ${lineNumber} 行 messages 第 ${messageIndex + 1} 条消息 content 不能为空。`,
            );
          }
        },
      );
      for (const field of ["chosen", "rejected"]) {
        const value = record[field];
        const validString = typeof value === "string" && value.trim();
        const validMessage =
          value &&
          typeof value === "object" &&
          !Array.isArray(value) &&
          (value as Record<string, unknown>).role === "assistant" &&
          typeof (value as Record<string, unknown>).content === "string" &&
          ((value as Record<string, unknown>).content as string).trim();
        if (!validString && !validMessage) {
          errors.push(
            `第 ${lineNumber} 行 ${field} 必须是非空字符串或 assistant 消息对象。`,
          );
        }
      }
      return;
    }

    if (!Array.isArray(record.messages) || record.messages.length < 2) {
      errors.push(
        `第 ${lineNumber} 行 messages 必须是至少包含 user 和 assistant 的数组。`,
      );
      return;
    }
    const messages = record.messages as Array<Record<string, unknown>>;
    const roles: string[] = [];
    messages.forEach((message, messageIndex) => {
      const role = message?.role;
      const content = message?.content;
      if (!["system", "user", "assistant"].includes(String(role))) {
        errors.push(
          `第 ${lineNumber} 行第 ${messageIndex + 1} 条消息 role 不支持。`,
        );
      }
      if (typeof content !== "string" || !content.trim()) {
        errors.push(
          `第 ${lineNumber} 行第 ${messageIndex + 1} 条消息 content 不能为空。`,
        );
      }
      roles.push(String(role));
    });
    const dialogueRoles = roles[0] === "system" ? roles.slice(1) : roles;
    if (dialogueRoles[0] !== "user") {
      errors.push(`第 ${lineNumber} 行对话必须从 user 开始，system 可选。`);
    }
    if (dialogueRoles.at(-1) !== "assistant") {
      errors.push(`第 ${lineNumber} 行必须以 assistant 回复结束。`);
    }
    dialogueRoles.forEach((role, roleIndex) => {
      const expected = roleIndex % 2 === 0 ? "user" : "assistant";
      if (role !== expected) {
        errors.push(`第 ${lineNumber} 行 user 与 assistant 消息必须交替出现。`);
      }
    });
  });

  if (!rows) errors.push("文件没有可用数据。 ");
  return { errors: errors.slice(0, 20), warnings, rows };
}
