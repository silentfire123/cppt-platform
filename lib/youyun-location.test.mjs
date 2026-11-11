import assert from "node:assert/strict";
import test from "node:test";

import { cachedYouyunRequest, clearYouyunRequestCache, friendlyYouyunError, loadYouyunSpecs, selectYouyunZone, youyunImageRuntimeDescription, youyunImageRuntimeLabel, youyunInventoryMessage, youyunMachineInventoryLabel, youyunRegionForSubmit, youyunRegionLabel, youyunRegionsForMode, youyunZonesForMode } from "./youyun-location.ts";

test("keeps unknown image metadata distinct from virtual-machine images", () => {
  assert.equal(youyunImageRuntimeLabel(true), "容器镜像");
  assert.equal(youyunImageRuntimeLabel(false), "虚机镜像");
  assert.equal(youyunImageRuntimeLabel(undefined), "镜像类型加载中");
  assert.equal(youyunImageRuntimeLabel(null), "镜像类型加载中");
  assert.equal(youyunImageRuntimeDescription(undefined), "正在确认镜像与实例类型");
});

test("drops a zone-shaped region before querying gpu-specs", async () => {
  const calls = [];
  const request = async (path) => {
    calls.push(path);
    return {
      region: "cn-wlcb",
      zone: "cn-wlcb-03",
      locationZone: "cn-wlcb-03",
      gpuType: "4090",
      defaultImageId: "train-img",
      machineTypes: [{ name: "4090", zone: "cn-wlcb-03", soldOut: false, trainingSupported: true }],
      capacity: [{ gpu: 1, cpu: 16, memoryGb: 64, available: true }],
    };
  };
  await loadYouyunSpecs(
    request,
    { gpuType: "", imageId: "", region: "cn-bj2-03", zone: "cn-wlcb-03", instanceMode: "pod" },
    "training",
    () => true,
  );
  const params = new URL(`http://local${calls[0]}`).searchParams;
  assert.equal(params.get("region"), null);
  assert.equal(params.get("zone"), "cn-wlcb-03");
});

test("youyunRegionForSubmit rejects zone codes posing as regions", () => {
  assert.equal(youyunRegionForSubmit("cn-bj2-03", "cn-wlcb-03"), "");
  assert.equal(youyunRegionForSubmit("cn-bj2-03", "cn-bj2-03"), "");
  assert.equal(youyunRegionForSubmit("cn-bj2", "cn-bj2-03"), "cn-bj2");
  assert.equal(youyunRegionForSubmit("cn-wlcb", "cn-wlcb-03"), "cn-wlcb");
  assert.equal(youyunRegionForSubmit("", "cn-wlcb-03"), "");
});

test("keeps the backend-selected zone-available GPU without requesting the first catalog item again", async () => {
  const calls = [];
  const request = async (path) => {
    calls.push(path);
    return {
      region: "cn-wlcb",
      zone: "cn-wlcb-03",
      locationZone: "cn-wlcb-03",
      gpuType: "4090",
      defaultImageId: "train-img",
      machineTypes: [
        { name: "3080Ti", zone: "cn-wlcb-02", soldOut: false, trainingSupported: true },
        { name: "4090", zone: "cn-wlcb-03", soldOut: false, trainingSupported: true },
      ],
      capacity: [{ gpu: 1, cpu: 16, memoryGb: 64, available: true }],
    };
  };
  const result = await loadYouyunSpecs(
    request,
    { gpuType: "", imageId: "", region: "cn-wlcb", zone: "cn-wlcb-03", instanceMode: "pod" },
    "training",
    () => true,
  );
  assert.equal(calls.length, 1);
  assert.equal(new URL(`http://local${calls[0]}`).searchParams.get("gpuType"), null);
  assert.equal(result.gpuType, "4090");
});

test("coalesces concurrent Youyun metadata requests and reuses the short TTL", async () => {
  let calls = 0;
  let release;
  const request = () => {
    calls += 1;
    return new Promise((resolve) => { release = resolve; });
  };
  const first = cachedYouyunRequest(request, "/youyun/gpu-specs?region=cn-wlcb", { ttlMs: 1000 });
  const second = cachedYouyunRequest(request, "/youyun/gpu-specs?region=cn-wlcb", { ttlMs: 1000 });
  assert.equal(calls, 1);
  release({ region: "cn-wlcb" });
  assert.deepEqual(await Promise.all([first, second]), [{ region: "cn-wlcb" }, { region: "cn-wlcb" }]);
  await cachedYouyunRequest(request, "/youyun/gpu-specs?region=cn-wlcb", { ttlMs: 1000 });
  assert.equal(calls, 1);
  clearYouyunRequestCache(request);
});

test("force reload bypasses a cached inventory snapshot", async () => {
  let calls = 0;
  const request = async () => {
    calls += 1;
    return {
      region: "cn-wlcb",
      zone: "cn-wlcb-03",
      gpuType: "4090",
      instanceMode: "pod",
      machineTypes: [{ name: "4090", zone: "cn-wlcb-03", soldOut: false }],
      capacity: [{ gpu: 1, cpu: 16, memoryGb: 64, available: calls > 1 }],
    };
  };
  const selection = {
    gpuType: "4090",
    imageId: "train-img",
    region: "cn-wlcb",
    zone: "cn-wlcb-03",
    instanceMode: "pod",
  };

  const first = await loadYouyunSpecs(request, selection, "training", () => true);
  const refreshed = await loadYouyunSpecs(
    request,
    selection,
    "training",
    () => true,
    { force: true },
  );

  assert.equal(first.capacity[0].available, false);
  assert.equal(refreshed.capacity[0].available, true);
  assert.equal(calls, 2);
});

const zones = [
  { region: "cn-bj2", zone: "cn-bj2-01", is_pod: false },
  { region: "cn-bj2", zone: "cn-bj2-03", is_pod: true },
  { region: "cn-sh2", zone: "cn-sh2-01", is_pod: false },
  { region: "cn-sh2", zone: "cn-sh2-02", is_pod: true },
];

test("keeps Pod region changes on a Pod zone", () => {
  assert.equal(selectYouyunZone(zones, "cn-sh2", "pod"), "cn-sh2-02");
});

test("keeps UHost region changes on a UHost zone", () => {
  assert.equal(selectYouyunZone(zones, "cn-sh2", "uhost"), "cn-sh2-01");
});

test("returns no zone when the selected mode is unavailable in a region", () => {
  assert.equal(selectYouyunZone(zones, "cn-none", "pod"), "");
});

test("derives zones for the requested instance form", () => {
  assert.deepEqual(
    youyunZonesForMode(zones, "pod").map((item) => item.zone),
    ["cn-bj2-03", "cn-sh2-02"],
  );
  assert.deepEqual(
    youyunZonesForMode(zones, "uhost").map((item) => item.zone),
    ["cn-bj2-01", "cn-sh2-01", "cn-sh2-02"],
  );
});

test("derives distinct regions for the requested instance form", () => {
  // The same region offering both forms stays present in each mode list.
  assert.deepEqual(youyunRegionsForMode(zones, "pod"), ["cn-bj2", "cn-sh2"]);
  assert.deepEqual(youyunRegionsForMode(zones, "uhost"), ["cn-bj2", "cn-sh2"]);
  const uhostOnly = [{ region: "cn-wlcb", zone: "cn-wlcb-03", is_pod: false }];
  assert.deepEqual(youyunRegionsForMode(uhostOnly, "uhost"), ["cn-wlcb"]);
  assert.deepEqual(youyunRegionsForMode(uhostOnly, "pod"), []);
});

test("uses region capabilities to resolve the matching zone for each mode", () => {
  const capabilities = [
    {
      region: "cn-wlcb",
      displayName: "华北二区",
      modes: { pod: "cn-wlcb-03", uhost: "cn-wlcb-01" },
      zones: { pod: ["cn-wlcb-03"], uhost: ["cn-wlcb-01"] },
    },
  ];
  assert.deepEqual(youyunRegionsForMode(zones, "pod", capabilities), ["cn-wlcb"]);
  assert.equal(selectYouyunZone(zones, "cn-wlcb", "pod", capabilities), "cn-wlcb-03");
  assert.equal(selectYouyunZone(zones, "cn-wlcb", "uhost", capabilities), "cn-wlcb-01");
  assert.deepEqual(youyunZonesForMode(zones, "pod", "cn-wlcb", capabilities), []);
});

test("keeps the current dual-mode region when switching modes", () => {
  const capabilities = [
    { region: "cn-wlcb", displayName: "华北二区", modes: { pod: "cn-wlcb-03", uhost: "cn-wlcb-01" } },
    { region: "cn-bj2", displayName: "华北一区", modes: { pod: "cn-bj2-03", uhost: null } },
  ];
  assert.deepEqual(youyunRegionsForMode(zones, "uhost", capabilities), ["cn-wlcb"]);
});

test("formats Chinese region labels and actionable provider errors", () => {
  assert.equal(youyunRegionLabel("cn-wlcb"), "华北二区");
  assert.equal(youyunRegionLabel("cn-unknown"), "cn-unknown");
  assert.equal(friendlyYouyunError("Params [ChargeType] not available", "price"), "库存查询成功，但当前规格暂时无法获取价格，请稍后重试。");
  assert.equal(friendlyYouyunError("Resource capacity is unavailable"), "当前地域暂无该 GPU 可用库存，请切换地域或 GPU。");
  assert.match(friendlyYouyunError("CPU memory capacity unavailable", "price"), /CPU \/ 内存组合暂无库存/);
  assert.match(friendlyYouyunError("ResourceEnough false", "price"), /GPU 可用库存/);
});

test("labels concrete Zone inventory without claiming unknown stock", () => {
  assert.equal(youyunMachineInventoryLabel({ soldOut: false, inventoryStatus: "available" }), "当前 GPU 规格有库存");
  assert.equal(youyunMachineInventoryLabel({ soldOut: true, inventoryStatus: "sold_out" }), "当前 GPU 规格暂无库存");
  assert.equal(youyunMachineInventoryLabel({ soldOut: true, inventoryStatus: "error" }), "库存查询失败，请刷新");
  assert.equal(youyunMachineInventoryLabel({ soldOut: true, inventoryStatus: "unknown" }), "库存待确认");
});

test("keeps GPU stock, CPU/memory stock and platform support separate", () => {
  const specs = { region: "cn-bj2", zone: "cn-bj2-03", gpuType: "3090", machineTypes: [{ name: "3090", zone: "cn-bj2-03", soldOut: true }], capacity: [] };
  assert.match(youyunInventoryMessage(specs, "training"), /暂无 GPU 可用库存/);
  specs.machineTypes[0].soldOut = false;
  assert.match(youyunInventoryMessage(specs, "inference"), /单卡 CPU \/ 内存组合暂无库存/);
  specs.machineTypes[0].inferenceSupported = false;
  assert.match(youyunInventoryMessage(specs, "inference"), /暂不支持平台推理/);
  specs.machineTypes[0].inferenceSupported = true;
  specs.capacity.push({ gpu: 1, cpu: 8, memoryGb: 32, available: true });
  assert.equal(youyunInventoryMessage(specs, "inference"), "");
});
