import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("model library tabs keep their state instead of remounting", () => {
  const source = readFileSync(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(source, /<ModelManagementModule\s+key=\{modelView\}/);
});

test("model library loads only the catalog needed by the visible tab", () => {
  const source = readFileSync(
    new URL("../components/model-management-module.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(source, /useEffect\(\(\) => \{\s*void refreshSupportingAssets\(\)/);
  assert.match(source, /if \(tab !== "trained"\) return;[\s\S]{0,180}void refreshTrainedAssets\(\)/);
  assert.match(source, /if \(tab !== "base"\) return;[\s\S]{0,180}void refreshPublicBaseModels\(\)/);
});
