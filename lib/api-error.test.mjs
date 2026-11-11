import assert from "node:assert/strict";
import test from "node:test";

import { formatApiError } from "./api-error.ts";

test("formats FastAPI validation details instead of rendering object objects", () => {
  assert.equal(
    formatApiError({
      detail: [
        { loc: ["body", "parameters", "maxSequenceLength"], msg: "Input should be less than or equal to 32768" },
        { loc: ["body", "youyunResource"], msg: "Field required" },
      ],
    }),
    "parameters.maxSequenceLength：Input should be less than or equal to 32768；youyunResource：Field required",
  );
});

test("keeps string API details readable", () => {
  assert.equal(formatApiError({ detail: "请选择已发布的数据集。" }), "请选择已发布的数据集。");
});

test("formats a single validation object and preserves the endpoint fallback", () => {
  assert.equal(
    formatApiError({ detail: { loc: ["body", "name"], msg: "Field required" } }),
    "name：Field required",
  );
  assert.equal(formatApiError({ detail: { unexpected: true } }, "接口暂不可用"), "请求参数：参数不合法");
  assert.equal(formatApiError({ message: "upstream failed" }, "接口暂不可用"), "接口暂不可用");
});
