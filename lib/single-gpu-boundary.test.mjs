import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

const trainingSource = readFileSync(new URL("../components/training-module.tsx", import.meta.url), "utf8");
const backendSource = readFileSync(new URL("../backend/app.py", import.meta.url), "utf8");

test("training UI submits a single-GPU production configuration", () => {
  assert.match(trainingSource, /gpuCount: 1/);
  assert.match(trainingSource, /distributedStrategy: "single"/);
  assert.doesNotMatch(trainingSource, /youyun-distributed-controls/);
});

test("backend exposes the temporary single-GPU boundary", () => {
  assert.match(backendSource, /validate_single_gpu_training/);
  assert.match(backendSource, /当前生产边界：暂时只允许单卡单进程训练/);
});
