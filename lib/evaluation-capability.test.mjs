import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("production capability hides unfinished evaluation navigation and page", () => {
  const navigation = readFileSync(
    new URL("../components/workspace-navigation.tsx", import.meta.url),
    "utf8",
  );
  const page = readFileSync(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(navigation, /evaluationEnabled/);
  assert.match(navigation, /module !== "evaluation" \|\| evaluationEnabled/);
  assert.match(page, /module === "evaluation" && evaluationEnabled/);
});
