import assert from "node:assert/strict";
import test from "node:test";

import { selectDeploymentArtifactId } from "./model-handoff.ts";

const artifacts = [
  { id: "task-1:best", taskId: "task-1", variant: "best", deployable: true },
  { id: "task-1:final", taskId: "task-1", variant: "final", deployable: true },
  { id: "task-2:best", taskId: "task-2", variant: "best", deployable: true },
];

test("selects the requested Adapter variant for a model-management handoff", () => {
  assert.equal(
    selectDeploymentArtifactId(artifacts, {
      taskId: "task-1",
      adapterVariant: "final",
    }),
    "task-1:final",
  );
});

test("falls back to Best and then Final when the requested variant is unavailable", () => {
  assert.equal(
    selectDeploymentArtifactId(artifacts, {
      taskId: "task-1",
    }),
    "task-1:best",
  );
  assert.equal(
    selectDeploymentArtifactId(
      [{ id: "task-3:final", taskId: "task-3", variant: "final", deployable: true }],
      { taskId: "task-3", adapterVariant: "best" },
    ),
    "task-3:final",
  );
});

test("does not select artifacts from another task or an unavailable artifact", () => {
  assert.equal(
    selectDeploymentArtifactId(
      [{ id: "task-1:best", taskId: "task-1", variant: "best", deployable: false }],
      { taskId: "task-1" },
    ),
    null,
  );
});
