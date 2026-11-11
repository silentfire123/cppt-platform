import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import * as location from "./youyun-location.ts";

// Execute the production React handlers with deferred HTTP responses and hook state.
// This retains their captured render values, which is essential to reproduce races.
function harness(kind, request) {
  const file = new URL(`../components/${kind}-module.tsx`, import.meta.url);
  const source = ts.createSourceFile(file.pathname, readFileSync(file, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const names = new Set(["refreshYouyunSpecs", "refreshYouyunCapacity"]);
  const declarations = [];
  let priceEffect;
  function visit(node) {
    if (ts.isFunctionDeclaration(node) && names.has(node.name?.text)) declarations.push(node.getText(source));
    if (ts.isVariableDeclaration(node) && names.has(node.name.getText(source))) declarations.push(`const ${node.getText(source)};`);
    if (ts.isCallExpression(node) && node.expression.getText(source) === "useEffect" && node.arguments[0]?.getText(source).includes('"/youyun/price"')) priceEffect = node.arguments[0].getText(source);
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.equal(declarations.length, 2);
  const state = {
    backendMode: true, environment: "youyun", selectedYouyunMode: "pod", selectedYouyunRegion: "old", selectedYouyunZone: "old-1",
    selectedYouyunGpuType: "3090", selectedYouyunCpuCores: 4, selectedYouyunMemoryGb: 16,
    selectedYouyunImageId: "kept-image", youyunSpecs: null, youyunResource: { old: true },
    youyunLoading: false, youyunError: "", youyunPriceError: "", youyunPriceLoading: false,
    form: { provider: "youyun", youyunGpuType: "3090", youyunCpu: "4", youyunMemory: "16", name: "kept-service" },
  };
  const scope = {
    ...location, ...state, apiRequest: request, inferenceRequest: request, useCallback: (fn) => fn,
    youyunSpecsRequestRef: { current: 0 }, youyunPriceRequestRef: { current: 0 },
  };
  for (const key of Object.keys(state)) scope[`set${key[0].toUpperCase()}${key.slice(1)}`] = (value) => {
    state[key] = typeof value === "function" ? value(state[key]) : value;
  };
  const compiled = ts.transpileModule(declarations.join("\n"), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText;
  const handlers = new Function(...Object.keys(scope), `${compiled}\nreturn { refreshYouyunSpecs, refreshYouyunCapacity };`)(...Object.values(scope));
  return { state, scope, ...handlers, runPrice: () => {
    assert.ok(priceEffect);
    const effect = ts.transpileModule(`const run = ${priceEffect};`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
    const render = { ...scope, ...state };
    return new Function(...Object.keys(render), `${effect}\nreturn run();`)(...Object.values(render));
  }, refresh: (region) => kind === "training"
    ? handlers.refreshYouyunSpecs("", region, `${region}-1`)
    : handlers.refreshYouyunSpecs("", "kept-image", region, `${region}-1`) };
}

function specs(region, overrides = {}) {
  return {
    region, zone: `${region}-1`, gpuType: "3090", instanceMode: "pod", chargeType: "Postpay",
    machineTypes: [{ name: "3090", zone: `${region}-1`, soldOut: false, graphicsMemoryGb: 24 }],
    capacity: [{ gpu: 1, cpu: 8, memoryGb: 32, available: true }], zones: [], ...overrides,
  };
}

for (const kind of ["training", "inference"]) {
  test(`${kind}: pending old price is ignored immediately when regional refresh starts`, async () => {
    let finishPrice;
    let finishSpecs;
    const h = harness(kind, (path) => path === "/youyun/price"
      ? new Promise((resolve) => { finishPrice = resolve; })
      : new Promise((resolve) => { finishSpecs = resolve; }));
    h.state.youyunSpecs = specs("old", { capacity: [{ gpu: 1, cpu: 4, memoryGb: 16, available: true }] });
    h.runPrice();
    const refresh = h.refresh("new");
    finishPrice({ old: true });
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(h.state.youyunResource, null);
    h.runPrice();
    assert.equal(h.state.youyunPriceLoading, false);
    finishSpecs(specs("new"));
    await refresh;
  });

  test(`${kind}: price failure retains valid inventory and uses a separate price error`, async () => {
    const h = harness(kind, async () => { throw new Error("Params [ChargeType] not available"); });
    h.state.youyunSpecs = specs("old", { capacity: [{ gpu: 1, cpu: 4, memoryGb: 16, available: true }] });
    h.runPrice();
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(h.state.youyunError, "");
    assert.match(h.state.youyunPriceError, /库存查询成功.*无法获取价格/);
    assert.equal(h.state.youyunSpecs.capacity[0].available, true);
    assert.equal(h.state.youyunPriceLoading, false);
  });

  test(`${kind}: switching region once commits new capacity and resets invalid CPU/memory`, async () => {
    const calls = [];
    const h = harness(kind, async (path) => { calls.push(path); return specs("new"); });
    await h.refresh("new");
    assert.equal(h.state.youyunSpecs.region, "new");
    assert.equal(h.state.selectedYouyunZone, "new-1");
    assert.equal(kind === "training" ? h.state.selectedYouyunGpuType : h.state.form.youyunGpuType, "3090");
    assert.equal(Number(kind === "training" ? h.state.selectedYouyunCpuCores : h.state.form.youyunCpu), 8);
    assert.equal(h.state.selectedYouyunImageId, "kept-image");
    assert.equal(calls.length, 1);
  });

  test(`${kind}: older regional result cannot replace the latest region`, async () => {
    let finishOld;
    const h = harness(kind, (path) => {
      const region = new URL(path, "http://local").searchParams.get("region");
      return region === "slow" ? new Promise((resolve) => { finishOld = resolve; }) : Promise.resolve(specs(region));
    });
    const old = h.refresh("slow");
    await h.refresh("latest");
    finishOld(specs("slow"));
    await old;
    assert.equal(h.state.youyunSpecs.region, "latest");
    assert.equal(h.state.selectedYouyunZone, "latest-1");
    assert.equal(h.state.youyunLoading, false);
  });

  test(`${kind}: empty regional catalog response automatically loads GPU capacity`, async () => {
    const calls = [];
    const h = harness(kind, async (path) => {
      const query = new URL(path, "http://local").searchParams;
      calls.push(query);
      return query.get("gpuType") ? specs("new") : specs("new", { gpuType: "", capacity: [] });
    });
    await h.refresh("new");
    assert.equal(calls.length, 2);
    assert.equal(h.state.youyunSpecs.capacity[0].cpu, 8);
    assert.equal(h.state.youyunSpecs.gpuType, "3090");
    assert.equal(h.state.youyunLoading, false);
  });

  test(`${kind}: stale capacity failure cannot clear the latest successful region`, async () => {
    let failOld;
    let capacityStarted;
    const started = new Promise((resolve) => { capacityStarted = resolve; });
    const h = harness(kind, async (path) => {
      const query = new URL(path, "http://local").searchParams;
      if (query.get("region") !== "slow") return specs("latest");
      if (!query.get("gpuType")) return specs("slow", { gpuType: "", capacity: [] });
      capacityStarted();
      return new Promise((_, reject) => { failOld = reject; });
    });
    const old = h.refresh("slow");
    await started;
    await h.refresh("latest");
    failOld(new Error("ResourceEnough failed"));
    await old;
    assert.equal(h.state.youyunSpecs.region, "latest");
    assert.equal(h.state.youyunError, "");
    assert.equal(h.state.youyunLoading, false);
  });

  test(`${kind}: selected catalog GPU loads capacity from its actual Pod zone`, async () => {
    const calls = [];
    const h = harness(kind, async (path) => {
      const query = new URL(path, "http://local").searchParams;
      calls.push(query);
      if (query.get("gpuType") === "H200") return specs("new", { zone: "new-2", gpuType: "H200" });
      return specs("new", {
        gpuType: "P40", capacity: [{ gpu: 1, cpu: 2, memoryGb: 8, available: true }],
        machineTypes: [
          { name: "P40", zone: "new-1", soldOut: false, trainingSupported: false, inferenceSupported: false },
          { name: "H200", zone: "new-2", soldOut: false },
        ],
      });
    });
    await h.refresh("new");
    assert.equal(calls.length, 2);
    assert.equal(calls[1].get("gpuType"), "H200");
    assert.equal(calls[1].get("zone"), "new-2");
    assert.equal(h.state.youyunSpecs.gpuType, "H200");
    assert.equal(h.state.youyunSpecs.zone, "new-2");
    assert.equal(Number(kind === "training" ? h.state.selectedYouyunCpuCores : h.state.form.youyunCpu), 8);
  });
}
