import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

for (const moduleName of ["training", "inference"]) {
  test(`${moduleName}: renders exact five-zone selection with provider diagnostics`, () => {
    const source = readFileSync(new URL(`../components/${moduleName}-module.tsx`, import.meta.url), "utf8");
    assert.match(source, /youyunLocations\.map/);
    assert.match(source, /item\.supported === false/);
    assert.match(source, /item\.disabledReason \|\| item\.describe/);
    assert.match(source, /youyunPreferredMode\(item/);
    assert.match(source, /disabledReason/);
    assert.doesNotMatch(source, /youyun-instance-mode/);
  });
}

test("training image selection derives the compatible instance mode", () => {
  const source = readFileSync(new URL("../components/training-module.tsx", import.meta.url), "utf8");
  assert.match(source, /setSelectedYouyunMode\(nextMode\)/);
  assert.match(source, /setSelectedYouyunMode\(nextMode\)/);
  assert.match(source, /selectedYouyunImageContainer/);
});

test("region clicks pass image, region, zone and mode in the declared argument order", () => {
  const training = readFileSync(new URL("../components/training-module.tsx", import.meta.url), "utf8");
  const inference = readFileSync(new URL("../components/inference-module.tsx", import.meta.url), "utf8");
  assert.match(training, /refreshYouyunSpecs\("", item\.region, item\.zone, selectedYouyunImageId, nextMode\)/);
  assert.match(inference, /refreshYouyunSpecs\("", selectedYouyunImageId, item\.region, item\.zone, nextMode\)/);
});


test("inference retry inventory keeps the service-selected region, zone and mode", () => {
  const source = readFileSync(new URL("../components/inference-module.tsx", import.meta.url), "utf8");
  const match = source.match(/async function loadRetrySpecs[\s\S]*?\n  }/);
  assert.ok(match, "loadRetrySpecs handler should exist");
  assert.match(match[0], /query\.set\("region"/);
  assert.match(match[0], /query\.set\("zone"/);
  assert.match(match[0], /query\.set\("instanceMode"/);
});
