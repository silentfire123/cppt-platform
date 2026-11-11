import assert from "node:assert/strict";
import test from "node:test";

import { inspectTextDataset } from "./dataset-validation.ts";

test("reports the physical JSONL line number after blank lines", () => {
  const valid = JSON.stringify({
    messages: [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi" },
    ],
  });
  const result = inspectTextDataset(`${valid}\n\nnot-json\n`, "SFT");

  assert.equal(result.rows, 2);
  assert.deepEqual(result.errors, ["第 3 行不是合法 JSON。"]);
});

test("does not count blank lines as dataset rows", () => {
  const result = inspectTextDataset("\n  \r\n", "DPO");

  assert.equal(result.rows, 0);
  assert.deepEqual(result.errors, ["文件没有可用数据。 "]);
});
