import assert from "node:assert/strict";
import test from "node:test";

import { trainingModelCatalog } from "./training-model-source.ts";

test("Youyun defaults to the public model catalog while image metadata is loading", () => {
  assert.equal(
    trainingModelCatalog({
      environment: "youyun",
      selectedImageContainer: undefined,
      specsContainer: true,
    }),
    "public",
  );
  assert.equal(
    trainingModelCatalog({
      environment: "youyun",
      selectedImageContainer: undefined,
      specsContainer: undefined,
    }),
    "public",
  );
});

test("only a confirmed Youyun VM image uses the remote repository fallback", () => {
  assert.equal(
    trainingModelCatalog({
      environment: "youyun",
      selectedImageContainer: false,
      specsContainer: false,
    }),
    "modelscope",
  );
  assert.equal(
    trainingModelCatalog({ environment: "local" }),
    "modelscope",
  );
});
