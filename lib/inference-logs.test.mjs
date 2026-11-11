import assert from "node:assert/strict";
import test from "node:test";

import { normalizeInferenceLogs } from "./inference-logs.ts";

test("converts structured runtime log items into display lines", () => {
  assert.deepEqual(
    normalizeInferenceLogs({
      serviceId: "svc-1",
      runId: "run-1",
      items: [
        {
          id: 1,
          created_at: "2026-08-29T09:00:00Z",
          level: "INFO",
          source: "backend",
          message: "优云推理实例已创建",
        },
      ],
    }),
    ["[09:00:00] 优云推理实例已创建"],
  );
});

test("cleans remote vLLM prefixes while keeping platform time", () => {
  assert.deepEqual(
    normalizeInferenceLogs({
      items: [
        {
          created_at: "2026-08-29T11:01:55+08:00",
          level: "INFO",
          source: "remote",
          message: "[0;36m(EngineCore_DP0 pid=462) [0;0m INFO 08-29 03:00:59 [default_loader.py:308] Loading weights",
        },
        {
          created_at: "2026-08-29T11:01:56+08:00",
          level: "WARN",
          source: "remote",
          message: "WARNING 08-29 03:01:00 [cache.py:10] Cache directory unavailable",
        },
      ],
    }),
    ["[11:01:55] Loading weights", "[11:01:56] 警告：Cache directory unavailable"],
  );
});

test("cleans legacy lines from the fallback logs endpoint", () => {
  assert.deepEqual(
    normalizeInferenceLogs({ logs: ["[11:01:55] [remote] [INFO] INFO 08-29 03:00:59 [loader.py:1] Ready"] }),
    ["[11:01:55] Ready"],
  );
});

test("keeps compatibility with legacy string logs", () => {
  assert.deepEqual(normalizeInferenceLogs({ logs: ["legacy line"] }), ["legacy line"]);
});

test("strips stacked remote and pid prefixes while keeping health and progress records", () => {
  assert.deepEqual(
    normalizeInferenceLogs({
      items: [
        {
          created_at: "2026-08-29T17:02:23+00:00",
          level: "INFO",
          source: "remote",
          message: "[remote] \x1B[0;36m(APIServer pid=701)\x1B[0;0m INFO 08-29 17:02:23 [serving_models.py:168] Loaded new LoRA adapter: name 'svc-1'",
        },
        {
          created_at: "2026-08-29T17:02:24+00:00",
          level: "INFO",
          source: "remote",
          message: "[remote] (APIServer pid=701) INFO:     127.0.0.1:47594 - \"GET /health HTTP/1.1\" 200 OK",
        },
        {
          created_at: "2026-08-29T17:02:25+00:00",
          level: "INFO",
          source: "remote",
          message: "[remote] Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   0%|          | 0/8 [00:00<?, ?it/s]",
        },
      ],
    }),
    [
      "[17:02:23] Loaded new LoRA adapter: name 'svc-1'",
      '[17:02:24] 127.0.0.1:47594 - "GET /health HTTP/1.1" 200 OK',
      "[17:02:25] Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 0%| | 0/8 [00:00<?, ?it/s]",
    ],
  );
});
