import assert from "node:assert/strict";
import test from "node:test";
import { trainingArtifactInfo } from "./training-artifact-presentation.ts";
test("manifest is a file inventory, never a final Adapter", () => {
  const info = trainingArtifactInfo({ type: "manifest", name: "manifest", path: "/work/manifest.json" }, {});
  assert.equal(info.label, "产物清单（JSON）");
  assert.doesNotMatch(info.description, /可用于对照/);
});
test("best and final retain their distinct model labels", () => {
  assert.equal(trainingArtifactInfo({ type: "adapter", variant: "final", path: "/output", name: "final" }, {}).label, "最终 LoRA Adapter");
  assert.equal(trainingArtifactInfo({ type: "adapter", variant: "best", path: "/best", name: "best" }, { selectionMetric: "eval_loss" }).label, "最佳 LoRA Adapter");
});
