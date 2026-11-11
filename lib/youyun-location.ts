export type YouyunRequest = <T>(path: string, options?: RequestInit) => Promise<T>;

type CacheOptions = {
  /** How long a successful metadata response may be reused. */
  ttlMs?: number;
  /** Skip an existing value but still coalesce with an in-flight request. */
  force?: boolean;
};

type CacheEntry = {
  expiresAt: number;
  value: unknown;
};

type RequestCache = {
  entries: Map<string, CacheEntry>;
  pending: Map<string, Promise<unknown>>;
};

// Keep this cache scoped to the request function. That prevents test doubles and
// separately configured API clients from sharing provider data accidentally,
// while still deduplicating calls made by one mounted module.
const caches = new WeakMap<YouyunRequest, RequestCache>();

function cacheFor(request: YouyunRequest): RequestCache {
  const existing = caches.get(request);
  if (existing) return existing;
  const created: RequestCache = { entries: new Map(), pending: new Map() };
  caches.set(request, created);
  return created;
}

/**
 * Read a Youyun catalog/capacity GET with request coalescing and a short TTL.
 *
 * `apiRequest` deliberately uses `cache: "no-store"` because normal API data
 * must remain fresh. Youyun metadata is different: two React effects can ask
 * for the same catalog while a render is settling, and provider metadata is
 * expensive to assemble. This helper keeps browser-level freshness explicit
 * without changing the shared API client's semantics or caching mutations.
 */
export function cachedYouyunRequest<T>(
  request: YouyunRequest,
  path: string,
  { ttlMs = 5_000, force = false }: CacheOptions = {},
): Promise<T> {
  const cache = cacheFor(request);
  const now = Date.now();
  const existing = cache.entries.get(path);
  if (!force && existing && existing.expiresAt > now) {
    return Promise.resolve(existing.value as T);
  }

  const pending = cache.pending.get(path);
  if (pending) return pending as Promise<T>;

  const next = request<T>(path).then((value) => {
    cache.entries.set(path, { value, expiresAt: Date.now() + Math.max(0, ttlMs) });
    return value;
  }).finally(() => {
    if (cache.pending.get(path) === next) cache.pending.delete(path);
  });
  cache.pending.set(path, next);
  return next;
}

/** Clear metadata cached by one API client, primarily for explicit retry flows and tests. */
export function clearYouyunRequestCache(request: YouyunRequest): void {
  const cache = caches.get(request);
  if (!cache) return;
  cache.entries.clear();
}

export const YOUYUN_IMAGES_CACHE_TTL_MS = 60_000;
export const YOUYUN_SPECS_CACHE_TTL_MS = 5_000;

export type YouyunLocationMode = "uhost" | "pod";

const DUAL_MODE_ZONES = new Set(["cn-wlcb-01", "cn-sh2-02"]);

/** Render provider image metadata without treating a pending response as a VM. */
export function youyunImageRuntimeLabel(container: boolean | null | undefined): string {
  if (container === true) return "容器镜像";
  if (container === false) return "虚机镜像";
  return "镜像类型加载中";
}

export function youyunImageRuntimeDescription(container: boolean | null | undefined): string {
  if (container === true) return "可使用优云公共模型库路径";
  if (container === false) return "使用 ModelScope 或自定义环境";
  return "正在确认镜像与实例类型";
}

export type YouyunZone = {
  region: string;
  zone: string;
  describe: string;
  is_pod: boolean;
  supportedModes?: YouyunLocationMode[];
};

export type YouyunLocationOption = {
  region: string;
  zone: string;
  actualZone?: string | null;
  displayName?: string;
  describe: string;
  isPod?: boolean;
  supported?: boolean;
  disabledReason?: string | null;
  unsupportedImageTypes?: string[];
  supportedModes?: YouyunLocationMode[];
};

export type YouyunRegionCapability = {
  region: string;
  displayName: string;
  modes: { pod: string | null; uhost: string | null };
  zones?: { pod?: string[]; uhost?: string[] };
};

const REGION_LABELS: Record<string, string> = {
  "cn-bj2": "华北一区",
  "cn-wlcb": "华北二区",
  "cn-sh2": "上海二区",
};

/** Keep provider region codes out of the primary UI label. */
export function youyunRegionLabel(region: string): string {
  const normalized = region.trim();
  return REGION_LABELS[normalized] ?? normalized;
}

export function youyunZoneModes(zone: Pick<YouyunZone, "zone" | "is_pod" | "supportedModes"> | Pick<YouyunLocationOption, "zone" | "isPod" | "supportedModes">): YouyunLocationMode[] {
  if (zone.supportedModes?.length) return zone.supportedModes;
  if (DUAL_MODE_ZONES.has(zone.zone)) return ["pod", "uhost"];
  return [("is_pod" in zone ? zone.is_pod : zone.isPod) ? "pod" : "uhost"];
}

/** Keep a dual-mode zone aligned with the selected image/runtime and current mode. */
export function youyunPreferredMode(
  zone: Pick<YouyunZone, "zone" | "is_pod" | "supportedModes"> | Pick<YouyunLocationOption, "zone" | "isPod" | "supportedModes">,
  imageContainer: boolean | null | undefined,
  currentMode: YouyunLocationMode = "pod",
): YouyunLocationMode {
  const modes = youyunZoneModes(zone);
  const imageMode = imageContainer === true ? "pod" : imageContainer === false ? "uhost" : undefined;
  if (imageMode && modes.includes(imageMode)) return imageMode;
  if (modes.includes(currentMode)) return currentMode;
  return modes[0] ?? "pod";
}

/** Zones that sell the requested instance form (Pod containers vs UHost VMs). */
export function youyunZonesForMode(
  zones: YouyunZone[],
  mode: YouyunLocationMode,
  region?: string,
  capabilities: YouyunRegionCapability[] = [],
): YouyunZone[] {
  const matching = zones.filter((item) => youyunZoneModes(item).includes(mode) && (!region || item.region === region));
  const capability = capabilities.find((item) => item.region === region);
  const preferredZones = capability?.zones?.[mode];
  if (!preferredZones?.length) return matching;
  const preferred = new Set(preferredZones);
  return matching.filter((item) => preferred.has(item.zone));
}

/** Distinct regions offering the requested instance form. */
export function youyunRegionsForMode(
  zones: YouyunZone[],
  mode: YouyunLocationMode,
  capabilities: YouyunRegionCapability[] = [],
): string[] {
  if (capabilities.length) {
    return capabilities
      .filter((item) => Boolean(item.modes?.[mode]))
      .map((item) => item.region);
  }
  return Array.from(new Set(youyunZonesForMode(zones, mode).map((item) => item.region)));
}

/** Convert provider failures into actionable messages for the resource form. */
export function friendlyYouyunError(message: string, kind: "capacity" | "price" = "capacity"): string {
  const value = message.trim();
  if (/CPU|内存|memory/i.test(value) && /库存|ResourceEnough|capacity|无可用/i.test(value)) {
    return "当前 GPU 的 CPU / 内存组合暂无库存，请选择其他规格或地域。";
  }
  if (/库存|ResourceEnough|\bcapacity\b|无可用/i.test(value)) {
    return "当前地域暂无该 GPU 可用库存，请切换地域或 GPU。";
  }
  if (/ChargeType|价格|Price/i.test(value) || kind === "price") {
    return "库存查询成功，但当前规格暂时无法获取价格，请稍后重试。";
  }
  if (/InstanceType/i.test(value)) {
    return "当前地域暂时无法查询 GPU 规格，请切换地域或刷新重试。";
  }
  return value || "优云 GPU 资源查询失败，请刷新后重试。";
}

type YouyunMachine = {
  name: string;
  zone: string;
  soldOut: boolean;
  inventoryStatus?: "available" | "sold_out" | "unknown" | "error";
  inventoryMessage?: string;
  inventoryError?: string;
  trainingSupported?: boolean;
  inferenceSupported?: boolean;
};

type YouyunSpecsResult = {
  region: string;
  zone: string;
  gpuType: string;
  defaultImageId?: string | null;
  machineTypes: YouyunMachine[];
  gpuCatalog?: YouyunMachine[];
  capacity: { gpu: number; cpu: number; memoryGb: number; available: boolean }[];
  locationZone?: string;
  locations?: YouyunLocationOption[];
};

/** Resolve catalog, GPU zone and capacity as one selection before updating the form. */
export async function loadYouyunSpecs<T extends YouyunSpecsResult>(
  request: YouyunRequest,
  selection: { gpuType: string; imageId: string; region: string; zone: string; instanceMode: YouyunLocationMode },
  purpose: "training" | "inference",
  isCurrent: () => boolean,
  { force = false }: { force?: boolean } = {},
): Promise<T | null> {
  const query = (gpuType: string, region: string, zone: string, imageId: string) => {
    const params = new URLSearchParams({ instanceMode: selection.instanceMode });
    // A stale form draft can hold a zone code (e.g. cn-bj2-03) in the region
    // field. Sending it makes the backend reject the request with a zone/
    // region mismatch; drop it and let the backend derive the region from zone.
    const regionToSend = youyunRegionForSubmit(region, zone);
    if (gpuType) params.set("gpuType", gpuType);
    if (imageId) params.set("imageId", imageId);
    if (regionToSend) params.set("region", regionToSend);
    if (zone) params.set("zone", zone);
    return `/youyun/gpu-specs?${params}`;
  };
  let specs = await cachedYouyunRequest<T>(
    request,
    query(selection.gpuType, selection.region, selection.zone, selection.imageId),
    { ttlMs: YOUYUN_SPECS_CACHE_TTL_MS, force },
  );
  if (!isCurrent()) return null;
  const catalog = specs.machineTypes?.length ? specs.machineTypes : (specs.gpuCatalog ?? []);
  const supported = (machine: YouyunMachine) => purpose === "training"
    ? machine.trainingSupported !== false : machine.inferenceSupported !== false;
  const available = (machine: YouyunMachine) => !machine.soldOut && supported(machine);
  // When the backend had to probe the region catalog, its returned gpuType
  // is already the first family accepted by the concrete placement Zone.
  // Prefer that validated result; otherwise the old client-side catalog
  // selection remains the fallback for legacy responses/tests.
  const backendSelected = specs.gpuType
    ? catalog.find((item) => item.name === specs.gpuType && available(item))
    : undefined;
  const machine = backendSelected
    ?? catalog.find((item) => item.name === selection.gpuType && available(item))
    ?? catalog.find(available);
  if (!machine) return { ...specs, gpuType: "", capacity: [] };
  const zone = specs.locations
    ? (specs.locationZone || selection.zone || specs.zone)
    : (machine.zone || specs.zone);
  if (specs.gpuType !== machine.name) {
    const capacity = await cachedYouyunRequest<T>(
      request,
      query(machine.name, specs.region, zone, selection.imageId || specs.defaultImageId || ""),
      { ttlMs: YOUYUN_SPECS_CACHE_TTL_MS, force },
    );
    if (!isCurrent()) return null;
    specs = { ...specs, ...capacity, machineTypes: specs.machineTypes, gpuCatalog: specs.gpuCatalog };
  }
  return specs;
}

// Provider region codes as known to this platform. Anything else shaped like
// a zone (cn-xxx-NN) is a zone code that leaked into the region field.
const KNOWN_YOUYUN_REGIONS = new Set(["cn-bj2", "cn-wlcb", "cn-sh2"]);

/**
 * Reject a region that is actually a zone code (e.g. cn-bj2-03) before it can
 * poison price queries and task submissions. The backend also self-heals such
 * input, so this is a UI-side fast path keeping error banners quiet.
 */
export function youyunRegionForSubmit(region: string, zone: string): string {
  if (!region) return region;
  if (region === zone) return "";
  if (KNOWN_YOUYUN_REGIONS.has(region)) return region;
  // Zone codes carry a trailing numeric suffix; region codes never do.
  return /-\d+$/.test(region) ? "" : region;
}

export function youyunMachineInventoryLabel(machine: {
  soldOut: boolean;
  inventoryStatus?: "available" | "sold_out" | "unknown" | "error";
  inventoryMessage?: string;
}): string {
  if (machine.inventoryMessage) return machine.inventoryMessage;
  if (machine.inventoryStatus === "error") return "库存查询失败，请刷新";
  if (machine.inventoryStatus === "unknown") return "库存待确认";
  if (machine.inventoryStatus === "sold_out") return "当前 GPU 规格暂无库存";
  if (machine.inventoryStatus === "available") return "当前 GPU 规格有库存";
  return machine.soldOut ? "当前 Zone 暂无库存" : "机型可用，组合库存待确认";
}

/** Disable only an explicit stock/error result; an unprobed catalog card stays selectable. */
export function youyunMachineIsUnavailable(
  machine: {
    soldOut: boolean;
    inventoryStatus?: "available" | "sold_out" | "unknown" | "error";
    trainingSupported?: boolean;
    inferenceSupported?: boolean;
  },
  purpose?: "training" | "inference",
): boolean {
  if (machine.soldOut || machine.inventoryStatus === "sold_out" || machine.inventoryStatus === "error") return true;
  if (purpose === "training" && machine.trainingSupported === false) return true;
  if (purpose === "inference" && machine.inferenceSupported === false) return true;
  return false;
}

export function youyunMachineInventoryClass(machine: {
  soldOut: boolean;
  inventoryStatus?: "available" | "sold_out" | "unknown" | "error";
}): "tag-ready" | "tag-off" | "tag-pending" {
  if (youyunMachineIsUnavailable(machine)) return "tag-off";
  if (machine.inventoryStatus === "available") return "tag-ready";
  return "tag-pending";
}

export function youyunInventoryMessage(specs: YouyunSpecsResult, purpose: "training" | "inference"): string {
  const catalog = specs.machineTypes?.length ? specs.machineTypes : (specs.gpuCatalog ?? []);
  if (!catalog.length || catalog.every((item) => item.soldOut)) {
    return "当前 Zone 暂无 GPU 可用库存，请切换地域。";
  }
  const supported = catalog.filter((item) => purpose === "training" ? item.trainingSupported !== false : item.inferenceSupported !== false);
  if (!supported.length) return `当前地域的 GPU 暂不支持平台${purpose === "training" ? "训练" : "推理"}环境。`;
  if (!supported.some((item) => !item.soldOut)) return "当前 Zone 兼容平台的 GPU 暂无库存，请切换地域。";
  if (!specs.capacity.some((item) => item.gpu === 1 && item.available)) {
    return "当前 GPU 的单卡 CPU / 内存组合暂无库存，请选择其他 GPU 或地域。";
  }
  return "";
}

/** Pick the first zone that matches both the selected region and instance mode. */
export function selectYouyunZone(
  zones: YouyunZone[],
  region: string,
  mode: YouyunLocationMode,
  capabilities: YouyunRegionCapability[] = [],
): string {
  const capability = capabilities.find((item) => item.region === region);
  return capability?.modes?.[mode]
    ?? youyunZonesForMode(zones, mode, region, capabilities)[0]?.zone
    ?? "";
}
