import assert from "node:assert/strict";
import test from "node:test";

import {
  getYouyunInferenceImageAdvice,
  selectRecommendedYouyunInferenceImageId,
  youyunInferenceImageVersion,
  youyunInferenceImageOptionLabel,
} from "./youyun-inference-image.ts";

// Deliberately avoids any specific version literal: the platform follows the
// provider catalog, so these tests must pass whichever versions it publishes.
const images = [
  { id: "older", name: "vLLM v0.24.0", tags: [] },
  { id: "newest", name: "vLLM v0.26.1", tags: [] },
  { id: "middle", name: "vLLM v0.25.1", tags: [] },
  { id: "5090", name: "vLLM v0.24.0-5090", tags: [] },
];

test("selects the newest general-purpose image for ordinary GPUs", () => {
  assert.equal(
    selectRecommendedYouyunInferenceImageId(images, "3080Ti", "older"),
    "newest",
  );
  const advice = getYouyunInferenceImageAdvice(images[1], "V100S");
  assert.equal(advice.recommended, false);
  assert.equal(advice.compatible, false);
  assert.match(advice.message, /计算能力 7\.0|V100S/);
});

test("selects the RTX 5090 image only when the chosen GPU is a 5090", () => {
  assert.equal(
    selectRecommendedYouyunInferenceImageId(images, "RTX 5090", "newest"),
    "5090",
  );
  assert.equal(getYouyunInferenceImageAdvice(images[3], "4090").compatible, false);
  assert.equal(getYouyunInferenceImageAdvice(images[3], "5090").recommended, true);
});

test("follows the provider catalog when it publishes a newer version", () => {
  const withNewerRelease = [
    ...images,
    { id: "next", name: "vLLM v0.27.0", tags: [] },
  ];
  assert.equal(
    selectRecommendedYouyunInferenceImageId(withNewerRelease, "4090", "newest"),
    "next",
  );
  // No acceptance warning is attached to a version the platform has not seen.
  const advice = getYouyunInferenceImageAdvice(withNewerRelease[4], "4090");
  assert.equal(advice.tone, "recommended");
  assert.doesNotMatch(advice.message, /验收/);
  assert.doesNotMatch(youyunInferenceImageOptionLabel(withNewerRelease[4], "4090"), /验收/);
});

test("prefers the backend recommendation over image name text", () => {
  const provided = [
    { id: "a", name: "vLLM 官方镜像", tags: ["v0.25.1"], compatibility: { recommendation: "general", version: "v0.25.1" } },
    { id: "b", name: "vLLM 官方镜像 5090", tags: ["v0.26.0"], compatibility: { recommendation: "rtx5090", version: "v0.26.0" } },
  ];
  assert.equal(selectRecommendedYouyunInferenceImageId(provided, "4090", null), "a");
  assert.equal(selectRecommendedYouyunInferenceImageId(provided, "5090", null), "b");
});

test("falls back to a general image for other GPUs when only 5090 and general exist", () => {
  assert.equal(
    selectRecommendedYouyunInferenceImageId(
      [images[3], images[2]],
      "4090",
      "5090",
    ),
    "middle",
  );
});

test("uses reliable compatibility or tag metadata for the image version", () => {
  assert.equal(
    youyunInferenceImageVersion({ id: "known", name: "vLLM image", tags: [], compatibility: { version: "0.25.1" } }),
    "0.25.1",
  );
  assert.equal(
    youyunInferenceImageVersion({ id: "tagged", name: "vLLM image", tags: ["v0.25.1"] }),
    "v0.25.1",
  );
  assert.equal(
    youyunInferenceImageVersion({ id: "unknown", name: "vLLM official image", tags: [] }),
    null,
  );
});
