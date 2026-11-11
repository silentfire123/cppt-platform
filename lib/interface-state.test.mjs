import assert from "node:assert/strict";
import test from "node:test";

import { shouldRestoreInterfaceState, restoreModelLibraryView } from "./interface-state.ts";

test("restores state on a browser reload", () => {
  assert.equal(shouldRestoreInterfaceState("reload"), true);
});

test("starts at overview on a fresh navigation from Control", () => {
  assert.equal(shouldRestoreInterfaceState("navigate"), false);
  assert.equal(shouldRestoreInterfaceState("back_forward"), false);
});

test("restores trained weights and tolerates legacy or invalid submenu state", () => {
  assert.equal(restoreModelLibraryView("trained"), "trained");
  for (const value of ["base", undefined, null, "archived", {}, 1]) {
    assert.equal(restoreModelLibraryView(value), "base");
  }
});
