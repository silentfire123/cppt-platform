import assert from "node:assert/strict";
import test from "node:test";

import { ApiRequestError, backendStateAfterSubmitError, isConnectivityError } from "./training-submit-state.ts";

test("treats transport failures as backend connectivity errors", () => {
  assert.equal(isConnectivityError(new ApiRequestError("训练服务连接失败", null)), true);
});

test("does not treat HTTP validation failures as backend connectivity errors", () => {
  const error = new ApiRequestError("parameters.epochs：Input should be greater than 0", 422);
  assert.equal(isConnectivityError(error), false);
});

test("does not treat provider business failures as backend connectivity errors", () => {
  const error = new ApiRequestError("优云 API CreateCompShareInstance 失败（RetCode=226645）：Service unavailable", 503);
  assert.equal(isConnectivityError(error), false);
});

test("keeps the current online state after a validation or provider failure", () => {
  assert.equal(
    backendStateAfterSubmitError("online", new ApiRequestError("参数不合法", 422)),
    "online",
  );
  assert.equal(
    backendStateAfterSubmitError("online", new ApiRequestError("优云服务暂不可用", 503)),
    "online",
  );
});
