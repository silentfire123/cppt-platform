import assert from "node:assert/strict";
import test from "node:test";

import { mergeTrainingDetail } from "./training-detail-state.ts";

test("a delayed detail response does not erase Loss received from the live stream", () => {
  const fetched = {
    id: "ft-live",
    state: "running",
    progress: 40,
    currentStep: "训练中",
    logs: ["fetched snapshot"],
    lossHistory: [],
  };
  const live = {
    ...fetched,
    logs: ["live log"],
    lossHistory: [{ step: 10, trainLoss: 1.23 }],
  };

  const merged = mergeTrainingDetail(live, fetched, true);

  assert.deepEqual(merged.logs, live.logs);
  assert.deepEqual(merged.lossHistory, live.lossHistory);
});

test("a detail response still supplies Loss when the live stream has none", () => {
  const fetched = {
    id: "ft-fetched",
    state: "completed",
    progress: 100,
    currentStep: "训练完成",
    logs: ["fetched snapshot"],
    lossHistory: [{ step: 20, trainLoss: 0.8 }],
  };
  const live = { ...fetched, logs: ["live log"], lossHistory: [] };

  const merged = mergeTrainingDetail(live, fetched, true);

  assert.deepEqual(merged.lossHistory, fetched.lossHistory);
});
