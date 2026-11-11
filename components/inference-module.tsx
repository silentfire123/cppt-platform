"use client";

import { useCapabilities, LOCAL_RUNTIME_DISABLED } from "../lib/use-capabilities";

import { type UIEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Plus, RefreshCw, X } from "lucide-react";
import { apiUrl, apiWebSocketUrl } from "../lib/api-config";
import { formatApiError } from "../lib/api-error";
import {
  selectDeploymentArtifactId,
  type InferenceCreateRequest,
} from "../lib/model-handoff";
import {
  getYouyunInferenceImageAdvice,
  selectRecommendedYouyunInferenceImageId,
  youyunInferenceImageVersion,
  youyunInferenceImageOptionLabel,
} from "../lib/youyun-inference-image";
import { normalizeInferenceLogs } from "../lib/inference-logs";
import { clearSessionDraft, readSessionDraft, INFERENCE_DRAFT_STORAGE_KEY, writeSessionDraft } from "../lib/interface-state";
import { cachedYouyunRequest, friendlyYouyunError, loadYouyunSpecs, YOUYUN_IMAGES_CACHE_TTL_MS, youyunInventoryMessage, youyunMachineInventoryClass, youyunMachineInventoryLabel, youyunMachineIsUnavailable, youyunPreferredMode, youyunRegionLabel, type YouyunLocationMode, type YouyunLocationOption } from "../lib/youyun-location";
import { BackButton } from "./back-button";
import { Pagination } from "./pagination";
import { useDialog } from "./use-dialog";

async function inferenceRequest<T>(
  path: string,
  options?: RequestInit,
  timeoutMs = 30000,
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const externalSignal = options?.signal;
  const abortFromCaller = () => controller.abort(externalSignal?.reason);
  externalSignal?.addEventListener("abort", abortFromCaller, { once: true });
  let response: Response;
  try {
    response = await fetch(apiUrl(path), {
      ...options,
      signal: controller.signal,
      credentials: "include",
      cache: "no-store",
    });
    } catch (error) {
    if (controller.signal.aborted && !externalSignal?.aborted) {
      throw new Error(`请求超时（${Math.round(timeoutMs / 1000)} 秒），请刷新后重试。`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
    externalSignal?.removeEventListener("abort", abortFromCaller);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(formatApiError(body, `推理服务接口请求失败（${response.status}）`));
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

type ServiceState =
  | "created"
  | "starting"
  | "running"
  | "stopping"
  | "stopped"
  | "recovering"
  | "failed"
  | "archived";
type ExposureState = "test" | "online" | "offline";
type InferenceView = "list" | "create" | "detail";
type DetailTab = "overview" | "debug" | "api" | "logs";
type LogConnectionState = "connecting" | "live" | "fallback";

type YouyunSpecs = {
  region: string;
  zone: string;
  instanceMode?: "uhost" | "pod";
  gpuType: string;
  gpuMemoryGb?: number | null;
  chargeType: string;
  machineTypes: Array<{
    name: string;
    zone: string;
    soldOut: boolean;
    description: string;
    graphicsMemoryGb: number | null;
    inventoryStatus?: "available" | "sold_out" | "unknown" | "error";
    inventoryMessage?: string;
    inventoryError?: string;
    inferenceSupported?: boolean;
    supportReason?: string | null;
  }>;
  gpuCatalog?: Array<{
    name: string;
    zone: string;
    soldOut: boolean;
    description: string;
    graphicsMemoryGb: number | null;
    priority?: number;
    compatible?: boolean;
    inferenceSupported?: boolean;
    supportReason?: string | null;
  }>;
  capacity: Array<{
    gpu: number;
    cpu: number;
    memoryGb: number;
    available: boolean;
  }>;
  zones: Array<{ region: string; zone: string; describe: string; is_pod: boolean }>;
  locations?: YouyunLocationOption[];
  locationZone?: string;
  imageType?: string | null;
  container?: boolean | null;
};

type YouyunResource = {
  mode: "manual" | "auto";
  region: string;
  zone: string;
  gpuType: string;
  gpuCount: number;
  cpuCores: number;
  memoryGb: number;
  gpuMemoryGb?: number | null;
  imageId?: string;
  imageName?: string | null;
  imageVersion?: string | null;
  chargeType: string;
  price: {
    currency: string;
    instance: number;
    disks: number;
    systemDisks: number;
    image: number;
    total: number;
  };
};

type YouyunImage = {
  id: string;
  name: string;
  type: string;
  status: string;
  tags: string[];
  description: string;
  source?: string;
  instanceMode?: "pod" | "uhost";
  container?: boolean | null;
  isDefault?: boolean;
  compatibility?: {
    version?: string | null;
    supportedGpu?: string[];
    recommendation?: "general" | "rtx5090" | string;
    recommendationReason?: string;
  };
};

type TrainingArtifact = {
  id: string;
  taskId: string;
  name: string;
  model: string;
  adapter: string;
  adapterPath: string;
  step: number;
  evalLoss: number | null;
  deployable: boolean;
  variant: "best" | "final";
};

function modelParameterLabel(model: string) {
  const match = model.match(/(?:^|[-_/])([0-9]+(?:\.[0-9]+)?)[bB](?:[-_/]|$)/);
  return match ? `${match[1]}B 参数规模` : "参数规模未从名称识别";
}

const YOUYUN_GPU_MEMORY_GB: Record<string, number> = {
  "2080": 8,
  "2080Ti": 11,
  "3080Ti": 12,
  "3090": 24,
  "4090": 24,
  "4090_48G": 48,
  "409048G": 48,
  "5090": 32,
  P40: 24,
  V100S: 32,
  H200: 141,
};

function youyunResourceSummary(resource: Record<string, unknown> | null | undefined) {
  if (!resource) return null;
  const gpuType = String(resource.gpuType ?? "优云 GPU");
  const gpuCount = Number(resource.gpuCount ?? 1);
  const cpuCores = Number(resource.cpuCores ?? 0);
  const memoryGb = Number(resource.memoryGb ?? 0);
  const vram = Number(resource.gpuMemoryGb ?? 0) || YOUYUN_GPU_MEMORY_GB[gpuType] || YOUYUN_GPU_MEMORY_GB[gpuType.replace(/[ _-]/g, "")];
  const vramLabel = vram ? ` · ${vram}GB 显存` : "";
  const imageId = String(resource.imageId ?? "").trim();
  const imageName = String(resource.imageName ?? "").trim() || imageId;
  const imageVersion = String(resource.imageVersion ?? "").trim();
  const imageLabel = imageId
    ? ` · 镜像 ${imageName} · ID ${imageId}（${imageVersion || "优云未提供版本信息"}）`
    : "";
  return `${gpuType}${vramLabel} · ${gpuCount} 卡 · ${cpuCores} 核 / ${memoryGb}GB 内存${imageLabel}`;
}

function youyunGpuHint(gpuType: string, memoryGb?: number | null) {
  const vram = memoryGb || YOUYUN_GPU_MEMORY_GB[gpuType] || YOUYUN_GPU_MEMORY_GB[gpuType.replace(/[ _-]/g, "")];
  if (!vram) return "显存信息以优云实时返回为准，建议先使用默认并发与上下文。";
  if (vram <= 12) return `${gpuType} 约 ${vram}GB 显存：建议保持并发不超过 4、显存利用率不高于 0.80。`;
  if (vram <= 24) return `${gpuType} 约 ${vram}GB 显存：可按基模能力逐步提高并发，首次启动建议保持默认值。`;
  return `${gpuType} 约 ${vram}GB 显存：可承载更长上下文或更高并发，仍以启动后的 vLLM 指标为准。`;
}

function formatServiceCreatedAt(value: string) {
  const match = value.match(/^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::(\d{2}))?/);
  if (match) return `${match[1]} ${match[2]}:${match[3] ?? "00"}`;
  return value.replace("T", " ").replace(/\.\d+(?:[+-]\d{2}:?\d{2}|Z)?$/, "").replace(/(?:[+-]\d{2}:?\d{2}|Z)$/, "");
}

type InferenceService = {
  id: string;
  name: string;
  description: string;
  state: ServiceState;
  exposure: ExposureState;
  model: string;
  adapter: string;
  taskId: string;
  gpu: string;
  memory: string;
  endpoint: string;
  createdAt: string;
  updatedAt: string;
  recoveryAttempts: number;
  error?: string;
  startupStage?: string | null;
  startupProgress?: number | null;
  startupStartedAt?: string | null;
  startupElapsedSeconds?: number | null;
  config?: {
    provider?: "local" | "youyun";
    youyunResource?: Record<string, unknown> | null;
    maxModelLength?: number;
    maxTokens?: number;
    modelContextLength?: number;
    modelContextSource?: string;
    temperature?: number;
    topP?: number;
    stream?: boolean;
    dtype?: string;
    quantization?: string;
    gpuMemoryUtilization?: number;
    kvCacheMode?: string;
    kvCacheMemoryGiB?: number | null;
    kvCacheDtype?: string;
    maxNumSeqs?: number;
    maxNumBatchedTokens?: number;
    enablePrefixCaching?: boolean;
    enableChunkedPrefill?: boolean;
    tensorParallelSize?: number;
    loadTimeout?: number;
    deploymentMode?: "managed" | "custom";
    customCommand?: string | null;
    customCommandFlags?: string[];
    adapterVariant?: "best" | "final";
    startup?: {
      stage?: string | null;
      message?: string | null;
      startedAt?: string | null;
      updatedAt?: string | null;
      elapsedSeconds?: number | null;
      error?: string | null;
    };
    remoteRuntime?: { state?: string } | null;
  };
};

type InferenceLogsResponse = {
  serviceId: string;
  runId?: string | null;
  firstItemId?: number;
  nextAfterId?: number;
  hasOlder?: boolean;
  hasMore?: boolean;
  logs?: string[];
  items?: Array<{
    id?: number;
    created_at?: string | null;
    level?: string | null;
    source?: string | null;
    message?: string | null;
  }>;
};

type BackendTrainingTask = {
  id: string;
  name: string;
  state: string;
  model: string;
  bestStep?: number | null;
  bestEvalLoss?: number | null;
  bestAdapterPath?: string | null;
  selectedAdapterPath?: string | null;
  selectedAdapterVariant?: "best" | "final" | null;
  finalStep?: number | null;
  finalEvalLoss?: number | null;
  artifacts?: Array<{
    type?: string;
    variant?: "best" | "final";
    checkpointStep?: number;
    evalLoss?: number;
    path?: string;
    selected?: boolean;
  }>;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  meta: string;
};

const artifacts: TrainingArtifact[] = [
  {
    id: "ft-260817-103208-a421:best",
    taskId: "ft-260817-103208-a421",
    name: "客服问答 LoRA 优化",
    model: "Qwen/Qwen2.5-3B-Instruct",
    adapter: "Best Adapter / checkpoint-420",
    adapterPath: "/workspace/training-work/ft-260817-103208-a421/checkpoint-420",
    step: 420,
    evalLoss: 0.9826,
    deployable: true,
    variant: "best",
  },
  {
    id: "ft-260816-164503-b197:best",
    taskId: "ft-260816-164503-b197",
    name: "产品知识助手 QLoRA",
    model: "Qwen/Qwen2.5-7B-Instruct",
    adapter: "Best Adapter / checkpoint-300",
    adapterPath: "/workspace/training-work/ft-260816-164503-b197/checkpoint-300",
    step: 300,
    evalLoss: 1.0418,
    deployable: true,
    variant: "best",
  },
  {
    id: "ft-260815-215401-88a5:final",
    taskId: "ft-260815-215401-88a5",
    name: "Qwen2.5-3B LoRA（历史任务）",
    model: "Qwen/Qwen2.5-3B-Instruct",
    adapter: "Final Adapter / checkpoint-909",
    adapterPath: "/workspace/training-work/ft-260815-215401-88a5/output",
    step: 909,
    evalLoss: 1.227,
    deployable: true,
    variant: "final",
  },
];

const initialServices: InferenceService[] = [
  {
    id: "svc-260817-001",
    name: "客服问答-正式服务",
    description: "面向内部客服工作台的稳定问答接口",
    state: "running",
    exposure: "online",
    model: "Qwen2.5-3B-Instruct",
    adapter: "Best Adapter · Step 420",
    taskId: artifacts[0].taskId,
    gpu: "GPU 0 · 1 卡",
    memory: "18.7 / 24 GB",
    endpoint: "/v1/chat/completions",
    createdAt: "2026-08-17 10:42",
    updatedAt: "刚刚",
    recoveryAttempts: 0,
  },
  {
    id: "svc-260816-003",
    name: "产品知识助手-测试",
    description: "用于产品知识数据集的回归测试",
    state: "stopped",
    exposure: "test",
    model: "Qwen2.5-7B-Instruct",
    adapter: "Best Adapter · Step 300",
    taskId: artifacts[1].taskId,
    gpu: "未占用",
    memory: "0 / 24 GB",
    endpoint: "/v1/chat/completions",
    createdAt: "2026-08-16 17:08",
    updatedAt: "昨天 18:20",
    recoveryAttempts: 0,
  },
  {
    id: "svc-260816-002",
    name: "售后分类助手",
    description: "验证售后问题分类与回复质量",
    state: "failed",
    exposure: "offline",
    model: "Qwen2.5-3B-Instruct",
    adapter: "Best Adapter · Step 260",
    taskId: "ft-260816-142233-c618",
    gpu: "未占用",
    memory: "0 / 24 GB",
    endpoint: "/v1/chat/completions",
    createdAt: "2026-08-16 14:38",
    updatedAt: "昨天 15:02",
    recoveryAttempts: 0,
    error: "CUDA out of memory：模型加载阶段显存不足。",
  },
  {
    id: "svc-260815-004",
    name: "FAQ 回归基线",
    description: "保留配置，等待手动启动",
    state: "created",
    exposure: "test",
    model: "Qwen2.5-3B-Instruct",
    adapter: "Best Adapter · Step 180",
    taskId: "ft-260815-112819-d201",
    gpu: "未占用",
    memory: "0 / 24 GB",
    endpoint: "/v1/chat/completions",
    createdAt: "2026-08-15 11:58",
    updatedAt: "2 天前",
    recoveryAttempts: 0,
  },
  {
    id: "svc-260814-005",
    name: "FAQ 历史回归",
    description: "已归档的历史测试服务，保留运行记录",
    state: "archived",
    exposure: "offline",
    model: "Qwen2.5-3B-Instruct",
    adapter: "Best Adapter · Step 180",
    taskId: "ft-260815-112819-d201",
    gpu: "未占用",
    memory: "0 / 24 GB",
    endpoint: "/v1/chat/completions",
    createdAt: "2026-08-14 09:20",
    updatedAt: "4 天前",
    recoveryAttempts: 0,
  },
];

function deployableArtifactsFromTasks(
  tasks: BackendTrainingTask[],
): TrainingArtifact[] {
  return tasks.flatMap((task) => {
    if (!["completed", "stopped"].includes(task.state)) return [];

    const byVariant = new Map<
      "best" | "final",
      NonNullable<BackendTrainingTask["artifacts"]>[number]
    >();
    for (const artifact of task.artifacts ?? []) {
      if (
        artifact.type === "adapter" &&
        (artifact.variant === "best" || artifact.variant === "final") &&
        artifact.path
      ) {
        byVariant.set(artifact.variant, artifact);
      }
    }

    if (task.bestAdapterPath && !byVariant.has("best")) {
      byVariant.set("best", {
        type: "adapter",
        variant: "best",
        path: task.bestAdapterPath,
        checkpointStep: task.bestStep ?? undefined,
        evalLoss: task.bestEvalLoss ?? undefined,
      });
    }
    if (
      task.selectedAdapterPath &&
      task.selectedAdapterVariant &&
      !byVariant.has(task.selectedAdapterVariant)
    ) {
      byVariant.set(task.selectedAdapterVariant, {
        type: "adapter",
        variant: task.selectedAdapterVariant,
        path: task.selectedAdapterPath,
      });
    }

    return (["best", "final"] as const).flatMap((variant) => {
      const artifact = byVariant.get(variant);
      if (!artifact?.path) return [];
      const step =
        artifact.checkpointStep ??
        (variant === "best" ? task.bestStep : task.finalStep) ??
        0;
      const evalLoss =
        artifact.evalLoss ??
        (variant === "best" ? task.bestEvalLoss : task.finalEvalLoss) ??
        null;
      const location = step > 0 ? `checkpoint-${step}` : "output";
      return [
        {
          id: `${task.id}:${variant}`,
          taskId: task.id,
          name: task.name,
          model: task.model,
          adapter: `${variant === "best" ? "Best" : "Final"} Adapter / ${location}`,
          adapterPath: artifact.path,
          step,
          evalLoss,
          deployable: true,
          variant,
        },
      ];
    });
  });
}

const stateMeta: Record<
  ServiceState,
  { label: string; className: string; hint: string }
> = {
  created: {
    label: "待启动",
    className: "inference-created",
    hint: "配置已保存，尚未占用 GPU",
  },
  starting: {
    label: "启动中",
    className: "inference-starting",
    hint: "正在加载基模与所选 Adapter",
  },
  running: {
    label: "运行中",
    className: "inference-running",
    hint: "服务已监听内部接口",
  },
  stopping: {
    label: "停止中",
    className: "inference-stopping",
    hint: "正在释放 vLLM 与 GPU 资源",
  },
  stopped: {
    label: "已停止",
    className: "inference-stopped",
    hint: "配置保留，GPU 已释放",
  },
  recovering: {
    label: "重试中",
    className: "inference-recovering",
    hint: "正在按原配置重新启动",
  },
  failed: {
    label: "启动失败",
    className: "inference-failed",
    hint: "可按原配置重试",
  },
  archived: {
    label: "已归档",
    className: "inference-archived",
    hint: "配置和记录保留，服务不可启动",
  },
};

const exposureMeta: Record<
  ExposureState,
  { label: string; className: string }
> = {
  test: { label: "测试中", className: "exposure-test" },
  online: { label: "已上线", className: "exposure-online" },
  offline: { label: "已下线", className: "exposure-offline" },
};

const activeStates: ServiceState[] = [
  "starting",
  "running",
  "stopping",
  "recovering",
];

const BACKEND_UNAVAILABLE_NOTICE =
  "推理服务后端连接中断，页面数据已暂停刷新，正在每 5 秒自动重连；重连成功后自动恢复实时数据。";

function serviceProvider(service: InferenceService): "local" | "youyun" {
  return service.config?.provider === "youyun" || service.config?.youyunResource
    ? "youyun"
    : "local";
}

function normalizeService(service: InferenceService): InferenceService {
  if (!["failed", "stopped", "archived"].includes(service.state)) {
    return service;
  }
  const cleanupPending = service.config?.remoteRuntime?.state === "cleanup_failed";
  return {
    ...service,
    exposure: "offline",
    gpu: cleanupPending ? "优云实例待清理" : "未占用",
    memory: cleanupPending ? "清理失败" : "0 / 0 GB",
  };
}

export function InferenceModule({
  createRequest,
  onCreateRequestHandled,
}: {
  createRequest?: InferenceCreateRequest | null;
  onCreateRequestHandled?: (requestId: number) => void;
}) {
  const { localRuntimeEnabled, capabilities } = useCapabilities();
  const handledCreateRequestRef = useRef<number | null>(null);
  const [view, setView] = useState<InferenceView>("list");
  const [draftHydrated, setDraftHydrated] = useState(false);
  const [services, setServices] = useState(initialServices);
  const [availableArtifacts, setAvailableArtifacts] =
    useState<TrainingArtifact[]>(artifacts);
  const [selectedId, setSelectedId] = useState(initialServices[0].id);
  const [tab, setTab] = useState<DetailTab>("overview");
  const [query, setQuery] = useState("");
  const [stateFilter, setStateFilter] = useState("all");
  const [notice, setNotice] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showConfigDetails, setShowConfigDetails] = useState(false);
  const [confirmAction, setConfirmAction] = useState<{
    type: "archive" | "delete";
    serviceId: string;
  } | null>(null);
  const [commandPreview, setCommandPreview] = useState<{
    command: string;
    flags: string[];
    model?: string;
    adapter?: string;
    port?: number;
    note?: string;
  } | null>(null);
  const [commandValidating, setCommandValidating] = useState(false);
  const [backendMode, setBackendMode] = useState(false);
  const [loadingServices, setLoadingServices] = useState(true);
  const [servicePage, setServicePage] = useState(1);
  const [servicePageInfo, setServicePageInfo] = useState({ pageSize: 5, total: 0, totalPages: 1 });
  const [storageBackend, setStorageBackend] = useState<"postgres" | "sqlite" | "unknown">("unknown");
  const [prompt, setPrompt] = useState("");
  const [sending, setSending] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatParameters, setChatParameters] = useState({
    maxTokens: 512,
    temperature: 0.7,
    topP: 0.9,
    stream: true,
  });
  const [modelContext, setModelContext] = useState<{
    maxContextLength: number;
    source: string;
    isFallback?: boolean;
  } | null>(null);
  const [modelContextLoading, setModelContextLoading] = useState(false);
  const [logsByService, setLogsByService] = useState<Record<string, string[]>>({});
  const [logsLoading, setLogsLoading] = useState<Record<string, boolean>>({});
  const [olderLogsLoading, setOlderLogsLoading] = useState<Record<string, boolean>>({});
  const [hasOlderLogs, setHasOlderLogs] = useState<Record<string, boolean>>({});
  const [logsError, setLogsError] = useState<Record<string, string>>({});
  const [logConnection, setLogConnection] = useState<LogConnectionState>("connecting");
  const logCursorRef = useRef<Record<string, number>>({});
  const logBeforeCursorRef = useRef<Record<string, number>>({});
  const logAtBottomRef = useRef<Record<string, boolean>>({});
  const logFetchPendingRef = useRef<Record<string, boolean>>({});
  const olderLogsPendingRef = useRef<Record<string, boolean>>({});
  const hasOlderLogsRef = useRef<Record<string, boolean>>({});
  const logViewerRef = useRef<HTMLPreElement | null>(null);
  const [youyunSpecs, setYouyunSpecs] = useState<YouyunSpecs | null>(null);
  const [selectedYouyunMode, setSelectedYouyunMode] = useState<YouyunLocationMode>("pod");
  const [selectedYouyunRegion, setSelectedYouyunRegion] = useState("");
  const [selectedYouyunZone, setSelectedYouyunZone] = useState("");
  const [youyunLoading, setYouyunLoading] = useState(false);
  const [youyunError, setYouyunError] = useState("");
  const [youyunResource, setYouyunResource] = useState<YouyunResource | null>(null);
  const [youyunPriceLoading, setYouyunPriceLoading] = useState(false);
  const [youyunPriceError, setYouyunPriceError] = useState("");
  const youyunSpecsRequestRef = useRef(0);
  const youyunPriceRequestRef = useRef(0);
  const [youyunImages, setYouyunImages] = useState<YouyunImage[]>([]);
  const [selectedYouyunImageId, setSelectedYouyunImageId] = useState("");
  const [youyunImagesLoading, setYouyunImagesLoading] = useState(false);
  const [youyunImagesError, setYouyunImagesError] = useState("");
  const [retryPicker, setRetryPicker] = useState<{
    serviceId: string;
    gpuType: string;
    cpu: string;
    memory: string;
    specs: YouyunSpecs | null;
    loading: boolean;
    error: string;
  } | null>(null);
  const retryDialogRef = useDialog<HTMLDivElement>(Boolean(retryPicker), useCallback(() => setRetryPicker(null), []));
  const confirmDialogRef = useDialog<HTMLDivElement>(Boolean(confirmAction), useCallback(() => setConfirmAction(null), []));
  const configDialogRef = useDialog<HTMLDivElement>(showConfigDetails, useCallback(() => setShowConfigDetails(false), []));
  const chatAbortRef = useRef<AbortController | null>(null);
  const chatThreadRef = useRef<HTMLDivElement | null>(null);
  const [form, setForm] = useState({
    name: "",
    description: "",
    artifactId: artifacts[0].id,
    maxLength: "4096",
    provider: "youyun" as "local" | "youyun",
    youyunGpuType: "4090",
    youyunCpu: "8",
    youyunMemory: "32",
    dtype: "auto",
    quantization: "none",
    kvCacheMode: "auto",
    kvCacheBytes: "8",
    kvCacheDtype: "auto",
    gpuMemory: "0.75",
    maxSequences: "4",
    batchTokens: "8192",
    prefixCaching: true,
    chunkedPrefill: true,
    timeout: "600",
    deploymentMode: "managed" as "managed" | "custom",
    customCommand: "",
  });

  useEffect(() => {
    if (capabilities && !localRuntimeEnabled && (form.provider === "local" || form.deploymentMode === "custom")) {
      setForm(current => ({ ...current, provider: "youyun", deploymentMode: "managed" }));
    }
  }, [capabilities, localRuntimeEnabled, form.provider, form.deploymentMode]);

  useEffect(() => {
    const navigationType = (window.performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined)?.type;
    if (navigationType === "reload") {
      const saved = readSessionDraft<Partial<{ version: number; view: InferenceView; selectedId: string; tab: DetailTab; form: Record<string, unknown> }>>(INFERENCE_DRAFT_STORAGE_KEY);
      if (saved?.version === 1 && saved.view) setView(saved.view);
      if (saved?.selectedId) setSelectedId(saved.selectedId);
      if (saved?.tab && ["overview", "debug", "api", "logs"].includes(saved.tab)) setTab(saved.tab);
      const savedForm = saved?.form;
      if (savedForm) {
        setForm((current) => ({
          ...current,
          ...Object.fromEntries(Object.entries(savedForm).filter(([key, value]) => key in current && value !== undefined)),
        }));
      }
    }
    setDraftHydrated(true);
  }, []);

  useEffect(() => {
    if (!draftHydrated) return;
    if (view !== "create") {
      clearSessionDraft(INFERENCE_DRAFT_STORAGE_KEY);
      return;
    }
    writeSessionDraft(INFERENCE_DRAFT_STORAGE_KEY, {
      version: 1,
      view,
      selectedId,
      tab,
      form,
    });
  }, [draftHydrated, view, selectedId, tab, form]);

  const leaveCreate = useCallback(() => {
    clearSessionDraft(INFERENCE_DRAFT_STORAGE_KEY);
    setView("list");
  }, []);

  const selectedService =
    services.find((service) => service.id === selectedId) ?? services[0];
  const chatOutputLimit = Math.max(
    1,
    Math.min(
      131072,
      Number(selectedService?.config?.modelContextLength) || 4096,
      Number(selectedService?.config?.maxModelLength) ||
        Number(selectedService?.config?.modelContextLength) ||
        4096,
    ),
  );
  const selectedArtifact =
    availableArtifacts.find((artifact) => artifact.id === form.artifactId) ??
    availableArtifacts[0];
  const selectedYouyunImage = youyunImages.find(
    (image) => image.id === selectedYouyunImageId,
  );
  const selectedYouyunImageAdvice = selectedYouyunImage
    ? getYouyunInferenceImageAdvice(selectedYouyunImage, form.youyunGpuType)
    : null;
  const youyunLocations = youyunSpecs?.locations ?? [];
  const selectedYouyunLocation = youyunLocations.find((item) => item.zone === (selectedYouyunZone || youyunSpecs?.locationZone));

  useEffect(() => {
    setChatParameters((current) => ({
      ...current,
      maxTokens: Math.min(current.maxTokens, chatOutputLimit),
    }));
  }, [chatOutputLimit, selectedService?.id]);

  useEffect(() => {
    if (tab !== "debug") return;
    const thread = chatThreadRef.current;
    if (!thread) return;
    const frame = window.requestAnimationFrame(() => {
      thread.scrollTop = thread.scrollHeight;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [messages, sending, tab, selectedId]);

  const contextLimit = modelContext?.maxContextLength ?? 0;
  const contextOptions = Array.from(
    new Set(
      modelContext
        ? [128, 512, 1024, 2048, 4096, 8192, 16384, 32768, contextLimit]
        : [128, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    ),
  )
    .filter((value) => !contextLimit || value <= contextLimit)
    .sort((left, right) => left - right);
  const activeLocalService = services.find(
    (service) =>
      serviceProvider(service) === "local" &&
      activeStates.includes(service.state),
  );
  const filteredServices = useMemo(
    () =>
      services.filter((service) => {
        const matchesQuery =
          service.name.includes(query) ||
          service.model.includes(query) ||
          service.taskId.includes(query);
        const matchesState =
          stateFilter === "all"
            ? service.state !== "archived"
            : service.state === stateFilter;
        return matchesQuery && matchesState;
      }),
    [query, services, stateFilter],
  );

  useEffect(() => {
    if (
      !createRequest ||
      handledCreateRequestRef.current === createRequest.requestId
    ) {
      return;
    }
    const artifactId = selectDeploymentArtifactId(
      availableArtifacts,
      createRequest,
    );
    if (!artifactId && loadingServices) return;

    handledCreateRequestRef.current = createRequest.requestId;
    setView("create");
    if (artifactId) {
      setForm((current) => ({ ...current, artifactId }));
      setNotice("");
    } else {
      setNotice(
        "没有找到该训练模型的有效 Best / Final Adapter，请刷新模型资产或确认训练产物仍然可用。",
      );
    }
    onCreateRequestHandled?.(createRequest.requestId);
  }, [
    availableArtifacts,
    createRequest,
    loadingServices,
    onCreateRequestHandled,
  ]);

  async function refreshServices() {
    setLoadingServices(true);
    try {
      const params = new URLSearchParams({ includeArchived: "true", page: String(servicePage), pageSize: "5" });
      if (query.trim()) params.set("query", query.trim());
      if (stateFilter !== "all") params.set("state", stateFilter);
      const result = await inferenceRequest<{ items: InferenceService[]; page: number; pageSize: number; total: number; totalPages: number }>(`/inference/services?${params}`);
      const normalizedServices = result.items.map(normalizeService);
      setServicePageInfo({ pageSize: result.pageSize, total: result.total, totalPages: result.totalPages });
      setServices(normalizedServices);
      setBackendMode(true);
      setNotice((current) =>
        current === BACKEND_UNAVAILABLE_NOTICE ? "" : current,
      );
      const health = await inferenceRequest<{ storageBackend?: string }>("/health").catch(() => null);
      setStorageBackend(health?.storageBackend === "postgres" ? "postgres" : health?.storageBackend === "sqlite" ? "sqlite" : "unknown");
      setSelectedId((current) =>
        normalizedServices.some((service) => service.id === current)
          ? current
          : (normalizedServices[0]?.id ?? ""),
      );
      try {
        const tasksResult = await inferenceRequest<{ items: BackendTrainingTask[] }>("/tasks?page=1&pageSize=100");
        const tasks = tasksResult.items;
        const nextArtifacts = deployableArtifactsFromTasks(tasks);
        setAvailableArtifacts(nextArtifacts);
        if (nextArtifacts.length) {
          setForm((current) => ({
            ...current,
            artifactId: nextArtifacts.some(
              (artifact) => artifact.id === current.artifactId,
            )
              ? current.artifactId
              : nextArtifacts[0].id,
          }));
        }
      } catch {
        setAvailableArtifacts([]);
      }
    } catch {
      setBackendMode(false);
      setStorageBackend("unknown");
      setNotice(BACKEND_UNAVAILABLE_NOTICE);
    } finally {
      setLoadingServices(false);
    }
  }

  useEffect(() => {
    setServicePage(1);
  }, [query, stateFilter]);

  useEffect(() => {
    if (backendMode) void refreshServices();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servicePage, query, stateFilter]);

  async function refreshYouyunCapacity(gpuType: string, imageId = selectedYouyunImageId, region = selectedYouyunRegion, zone = selectedYouyunZone) {
    if (!backendMode || !gpuType) return;
    await refreshYouyunSpecs(gpuType, imageId, region, zone, selectedYouyunMode, true);
  }


  async function refreshYouyunSpecs(
    gpuType = form.youyunGpuType,
    imageId = selectedYouyunImageId,
    region = selectedYouyunRegion,
    zone = selectedYouyunZone,
    mode: YouyunLocationMode = selectedYouyunMode,
    force = false,
  ) {
    if (!backendMode) return;
    const requestId = ++youyunSpecsRequestRef.current;
    ++youyunPriceRequestRef.current;
    setYouyunLoading(true);
    setYouyunError("");
    setYouyunResource(null);
    setYouyunPriceError("");
    setYouyunPriceLoading(false);
    setForm((current) => ({ ...current, youyunCpu: "", youyunMemory: "" }));
    try {
      const specs = await loadYouyunSpecs<YouyunSpecs>(inferenceRequest, {
        gpuType, imageId, region, zone, instanceMode: mode,
      }, "inference", () => requestId === youyunSpecsRequestRef.current, { force });
      if (!specs || requestId !== youyunSpecsRequestRef.current) return;
      setYouyunSpecs(specs);
      setSelectedYouyunMode(specs.instanceMode ?? (specs.zones.find((item) => item.zone === specs.zone)?.is_pod ? "pod" : "uhost"));
      setSelectedYouyunRegion(specs.region);
      setSelectedYouyunZone(specs.locationZone ?? specs.zone);
      const available = specs.capacity.filter((item) => item.gpu === 1 && item.available);
      const selected = available.find(
        (item) => item.cpu === Number(form.youyunCpu) && item.memoryGb === Number(form.youyunMemory),
      ) ?? available[0];
      setForm((current) => ({
        ...current,
        youyunGpuType: specs.gpuType,
        youyunCpu: selected ? String(selected.cpu) : "",
        youyunMemory: selected ? String(selected.memoryGb) : "",
      }));
    } catch (error) {
      if (requestId === youyunSpecsRequestRef.current) setYouyunError(friendlyYouyunError(error instanceof Error ? error.message : "", "capacity"));
    } finally {
      if (requestId === youyunSpecsRequestRef.current) setYouyunLoading(false);
    }
  }

  async function refreshYouyunImages(refreshSpecs = true) {
    if (!backendMode) return;
    setYouyunImagesLoading(true);
    setYouyunImagesError("");
    try {
      const result = await cachedYouyunRequest<{ images: YouyunImage[]; defaultImageId: string | null }>(
        inferenceRequest,
        "/youyun/inference-images",
        { ttlMs: YOUYUN_IMAGES_CACHE_TTL_MS },
      );
      setYouyunImages(result.images);
      const next = selectedYouyunImageId && result.images.some((item) => item.id === selectedYouyunImageId)
        ? selectedYouyunImageId
        : selectRecommendedYouyunInferenceImageId(
            result.images,
            form.youyunGpuType,
            result.defaultImageId,
          );
      if (next !== selectedYouyunImageId) {
        setSelectedYouyunImageId(next);
        setYouyunResource(null);
      }
      if (next && refreshSpecs) void refreshYouyunSpecs("", next);
    } catch (error) {
      setYouyunImagesError(error instanceof Error ? error.message : "优云官方 vLLM 镜像查询失败。");
    } finally {
      setYouyunImagesLoading(false);
    }
  }

  useEffect(() => {
    if (form.provider !== "youyun" || !backendMode) {
      ++youyunSpecsRequestRef.current;
      ++youyunPriceRequestRef.current;
      setYouyunLoading(false);
      setYouyunPriceLoading(false);
      setYouyunSpecs(null);
      setYouyunResource(null);
      setYouyunError("");
      return;
    }
    void refreshYouyunImages(false);
    // Load inventory independently so a slow image catalog cannot hide GPU
    // capacity (and vice versa).
    void refreshYouyunSpecs("", selectedYouyunImageId);
    return () => {
      ++youyunSpecsRequestRef.current;
      ++youyunPriceRequestRef.current;
    };
    // Provider 切换时读取一次实时库存；规格按钮会按 GPU 型号主动刷新。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form.provider, backendMode]);

  useEffect(() => {
    const requestId = ++youyunPriceRequestRef.current;
    setYouyunResource(null);
    setYouyunPriceError("");
    if (
      !backendMode ||
      form.provider !== "youyun" ||
      !youyunSpecs ||
      youyunLoading ||
      youyunError ||
      !selectedYouyunImageId ||
      !form.youyunGpuType ||
      !form.youyunCpu ||
      !form.youyunMemory ||
      !youyunSpecs.capacity.some((item) => item.gpu === 1 && item.available && item.cpu === Number(form.youyunCpu) && item.memoryGb === Number(form.youyunMemory))
    ) {
      setYouyunPriceLoading(false);
      return;
    }
    let cancelled = false;
    setYouyunPriceLoading(true);
    void inferenceRequest<YouyunResource>("/youyun/price", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "manual",
        purpose: "inference",
        gpuType: form.youyunGpuType,
        gpuCount: 1,
        cpuCores: Number(form.youyunCpu),
        memoryGb: Number(form.youyunMemory),
        imageId: selectedYouyunImageId || undefined,
        region: selectedYouyunRegion || youyunSpecs.region,
        zone: selectedYouyunZone || youyunSpecs.zone,
        instanceMode: selectedYouyunMode,
      }),
    })
      .then((resource) => {
        if (!cancelled && requestId === youyunPriceRequestRef.current) setYouyunResource(resource);
      })
      .catch((error) => {
        if (!cancelled && requestId === youyunPriceRequestRef.current) {
          setYouyunResource(null);
          setYouyunPriceError(friendlyYouyunError(error instanceof Error ? error.message : "", "price"));
        }
      })
      .finally(() => {
        if (!cancelled && requestId === youyunPriceRequestRef.current) setYouyunPriceLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [backendMode, form.provider, form.youyunGpuType, form.youyunCpu, form.youyunMemory, selectedYouyunImageId, selectedYouyunRegion, selectedYouyunZone, selectedYouyunMode, youyunSpecs, youyunLoading, youyunError]);

  useEffect(() => {
    void refreshServices();
    return () => chatAbortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (!selectedArtifact || !backendMode) {
      setModelContext(null);
      setModelContextLoading(false);
      return;
    }
    let cancelled = false;
    setModelContextLoading(true);
    setModelContext(null);
    void inferenceRequest<{
      maxContextLength: number;
      source: string;
      isFallback?: boolean;
    }>(`/inference/model-context?model=${encodeURIComponent(selectedArtifact.model)}&taskId=${encodeURIComponent(selectedArtifact.taskId)}&adapterVariant=${encodeURIComponent(selectedArtifact.variant)}`)
      .then((metadata) => {
        if (cancelled) return;
        setModelContext(metadata);
        const nextContext = Math.min(Number(form.maxLength), metadata.maxContextLength);
        setForm((current) => ({
          ...current,
          maxLength: String(nextContext),
        }));
      })
      .catch(() => {
        if (!cancelled) setModelContext(null);
      })
      .finally(() => {
        if (!cancelled) setModelContextLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [
    backendMode,
    selectedArtifact?.model,
    selectedArtifact?.taskId,
    selectedArtifact?.variant,
  ]);

  async function requestServiceAction<T>(
    id: string,
    action: string,
    options?: RequestInit,
  ) {
    const service = await inferenceRequest<T>(
      `/inference/services/${id}${action}`,
      options,
    );
    if (service && typeof service === "object")
      updateService(id, service as Partial<InferenceService>);
    return service;
  }

  async function fetchServiceLogs(serviceId: string, silent = false) {
    if (logFetchPendingRef.current[serviceId]) return;
    logFetchPendingRef.current[serviceId] = true;
    if (!silent) setLogsLoading((current) => ({ ...current, [serviceId]: true }));
    setLogsError((current) => ({ ...current, [serviceId]: "" }));
    try {
      const afterId = logCursorRef.current[serviceId] ?? 0;
      const shouldFollowLatest = logAtBottomRef.current[serviceId] ?? true;
      const payload = await inferenceRequest<InferenceLogsResponse>(
        `/inference/services/${serviceId}/logs?limit=200&afterId=${afterId}`,
      );
      const nextLogs = normalizeInferenceLogs(payload);
      const supportsIncremental = typeof payload.nextAfterId === "number";
      setLogsByService((current) => ({
        ...current,
        [serviceId]: afterId > 0 && supportsIncremental
          ? [...(current[serviceId] ?? []), ...nextLogs]
          : nextLogs,
      }));
      if (supportsIncremental) {
        logCursorRef.current[serviceId] = payload.nextAfterId ?? afterId;
      }
      if (afterId <= 0) {
        logBeforeCursorRef.current[serviceId] = payload.firstItemId ?? 0;
        hasOlderLogsRef.current[serviceId] = Boolean(payload.hasOlder);
        setHasOlderLogs((current) => ({
          ...current,
          [serviceId]: Boolean(payload.hasOlder),
        }));
      }
      if (nextLogs.length && (afterId <= 0 || shouldFollowLatest)) {
        window.requestAnimationFrame(() => {
          const viewer = logViewerRef.current;
          if (viewer) viewer.scrollTop = viewer.scrollHeight;
        });
      }
    } catch (error) {
      // Background polling is best-effort. Keep the last good log view and
      // only surface errors when the user explicitly presses Refresh.
      if (!silent) {
        setLogsError((current) => ({
          ...current,
          [serviceId]: error instanceof Error ? error.message : "读取运行日志失败。",
        }));
      }
    } finally {
      logFetchPendingRef.current[serviceId] = false;
      if (!silent) setLogsLoading((current) => ({ ...current, [serviceId]: false }));
    }
  }

  async function loadOlderServiceLogs(serviceId: string) {
    if (olderLogsPendingRef.current[serviceId]) return;
    if (!hasOlderLogsRef.current[serviceId]) return;
    const beforeId = logBeforeCursorRef.current[serviceId] ?? 0;
    if (beforeId <= 0) return;

    olderLogsPendingRef.current[serviceId] = true;
    setOlderLogsLoading((current) => ({ ...current, [serviceId]: true }));
    const viewer = logViewerRef.current;
    const previousScrollHeight = viewer?.scrollHeight ?? 0;
    const previousScrollTop = viewer?.scrollTop ?? 0;
    try {
      const payload = await inferenceRequest<InferenceLogsResponse>(
        `/inference/services/${serviceId}/logs?limit=200&beforeId=${beforeId}`,
      );
      const olderLogs = normalizeInferenceLogs(payload);
      if (olderLogs.length) {
        setLogsByService((current) => ({
          ...current,
          [serviceId]: [...olderLogs, ...(current[serviceId] ?? [])],
        }));
      }
      logBeforeCursorRef.current[serviceId] = payload.firstItemId ?? beforeId;
      hasOlderLogsRef.current[serviceId] = Boolean(payload.hasOlder);
      setHasOlderLogs((current) => ({
        ...current,
        [serviceId]: Boolean(payload.hasOlder),
      }));
      if (olderLogs.length) {
        window.requestAnimationFrame(() => {
          window.requestAnimationFrame(() => {
            const currentViewer = logViewerRef.current;
            if (!currentViewer) return;
            currentViewer.scrollTop =
              previousScrollTop + currentViewer.scrollHeight - previousScrollHeight;
          });
        });
      }
    } catch (error) {
      setLogsError((current) => ({
        ...current,
        [serviceId]: error instanceof Error ? error.message : "读取更早日志失败。",
      }));
    } finally {
      olderLogsPendingRef.current[serviceId] = false;
      setOlderLogsLoading((current) => ({ ...current, [serviceId]: false }));
    }
  }

  function handleLogScroll(serviceId: string, event: UIEvent<HTMLPreElement>) {
    const viewer = event.currentTarget;
    logAtBottomRef.current[serviceId] =
      viewer.scrollHeight - viewer.scrollTop - viewer.clientHeight <= 24;
    if (viewer.scrollTop <= 24) void loadOlderServiceLogs(serviceId);
  }

  function downloadServiceLogs(serviceId: string) {
    void (async () => {
      try {
        const response = await fetch(apiUrl(`/inference/services/${serviceId}/logs/download`), {
          credentials: "include",
          cache: "no-store",
        });
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(formatApiError(body, `下载运行日志失败（${response.status}）`));
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `${serviceId}-runtime.log`;
        anchor.click();
        URL.revokeObjectURL(url);
      } catch (error) {
        setLogsError((current) => ({
          ...current,
          [serviceId]: error instanceof Error ? error.message : "下载运行日志失败。",
        }));
      }
    })();
  }

  useEffect(() => {
    const serviceId = selectedService?.id;
    if (!serviceId) return;
    logCursorRef.current[serviceId] = 0;
    logBeforeCursorRef.current[serviceId] = 0;
    logAtBottomRef.current[serviceId] = true;
    hasOlderLogsRef.current[serviceId] = false;
    setLogsByService((current) => ({ ...current, [serviceId]: [] }));
    setHasOlderLogs((current) => ({ ...current, [serviceId]: false }));
  }, [selectedService?.id]);

  useEffect(() => {
    if (!backendMode || !selectedService) return;
    const shouldPoll = activeStates.includes(selectedService.state) || tab === "logs";
    if (!shouldPoll) return;
    setLogConnection("connecting");
    const serviceId = selectedService.id;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let fallbackTimer: number | undefined;
    let cancelled = false;
    let snapshotLogReceived = false;
    let terminal = false;
    const stopFallback = () => {
      if (fallbackTimer !== undefined) window.clearInterval(fallbackTimer);
      fallbackTimer = undefined;
    };
    const startFallback = () => {
      if (fallbackTimer !== undefined || cancelled) return;
      setLogConnection("fallback");
      fallbackTimer = window.setInterval(() => {
        void refreshServices();
        void fetchServiceLogs(serviceId, true);
      }, 5000);
    };
    const connect = () => {
      if (cancelled) return;
      const afterId = logCursorRef.current[serviceId] ?? 0;
      try {
        socket = new WebSocket(apiWebSocketUrl(`/inference/services/${serviceId}/stream?afterId=${afterId}`));
      } catch {
        startFallback();
        reconnectTimer = window.setTimeout(connect, 3000);
        return;
      }
      socket.onopen = () => {
        stopFallback();
        setLogConnection("live");
      };
      socket.onmessage = (event) => {
        let payload: any;
        try { payload = JSON.parse(event.data); } catch { return; }
        if (payload.type === "log") {
          const id = Number(payload.id ?? 0);
          if (id > 0 && id <= (logCursorRef.current[serviceId] ?? 0) && !payload.snapshot) return;
          const isSnapshotStart = Boolean(payload.snapshot) && !snapshotLogReceived;
          if (isSnapshotStart) {
            snapshotLogReceived = true;
            setLogsByService((current) => ({ ...current, [serviceId]: [] }));
          }
          const line = normalizeInferenceLogs({ items: [{ id, created_at: payload.createdAt, level: payload.level, source: payload.source, message: payload.message }] });
          if (line.length) setLogsByService((current) => ({ ...current, [serviceId]: [...(current[serviceId] ?? []), ...line] }));
          if (id > 0) logCursorRef.current[serviceId] = Math.max(logCursorRef.current[serviceId] ?? 0, id);
          return;
        }
        if (payload.type === "state" || payload.type === "terminal") {
          if (payload.service && typeof payload.service === "object") updateService(serviceId, payload.service);
          if (payload.snapshot && !snapshotLogReceived) setLogsByService((current) => ({ ...current, [serviceId]: [] }));
          if (payload.type === "terminal") {
            terminal = true;
            void refreshServices();
          }
        }
      };
      socket.onerror = startFallback;
      socket.onclose = () => {
        if (cancelled || terminal) return;
        startFallback();
        reconnectTimer = window.setTimeout(connect, 2500);
      };
    };
    void refreshServices();
    void fetchServiceLogs(serviceId, true).finally(() => {
      if (!cancelled) connect();
    });
    return () => {
      cancelled = true;
      stopFallback();
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      socket?.close();
      socket = null;
    };
    // Polling is intentionally tied to the selected service lifecycle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendMode, selectedService?.id, selectedService?.state, tab]);

  useEffect(() => {
    if (backendMode) return;
    // 后端连接中断时进入恢复探测：每 5 秒重试刷新，成功后 refreshServices
    // 会自动切回真实数据模式并清除横幅，页面不再冻结在最后一次快照上。
    const recovery = window.setInterval(() => {
      void refreshServices();
    }, 5000);
    return () => window.clearInterval(recovery);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendMode]);

  function updateService(id: string, patch: Partial<InferenceService>) {
    setServices((current) =>
      current.map((service) =>
        service.id === id
          ? { ...service, ...patch, updatedAt: "刚刚" }
          : service,
      ),
    );
  }

  function openDetail(id: string, nextTab: DetailTab = "overview") {
    setSelectedId(id);
    setTab(nextTab);
    setNotice("");
    setView("detail");
  }

  async function startService(id: string) {
    const target = services.find((service) => service.id === id);
    const conflict =
      target && serviceProvider(target) === "local"
        ? services.find(
            (service) =>
              service.id !== id &&
              serviceProvider(service) === "local" &&
              activeStates.includes(service.state),
          )
        : undefined;
    if (conflict) {
      setNotice(
        "本机 GPU 当前由“" + conflict.name + "”占用，请先停止该本地服务。",
      );
      return;
    }
    setNotice(
      `正在启动 vLLM：加载基模、挂载 ${selectedArtifact.variant === "best" ? "Best" : "Final"} Adapter，并执行健康检查。`,
    );
    updateService(id, {
      state: "starting",
      exposure: "test",
      gpu: "GPU 0 · 分配中",
      memory: "加载中",
    });
    if (backendMode) {
      try {
        const started = await requestServiceAction<InferenceService>(id, "/start", {
          method: "POST",
        });
        updateService(id, normalizeService(started));
        setNotice(
          started?.state === "running"
            ? "vLLM 健康检查通过，服务已运行。"
            : "优云启动任务已提交，正在后台创建实例并加载模型；可在运行日志查看进度。",
        );
      } catch (error) {
        setNotice(
          error instanceof Error ? error.message : "启动推理服务失败。",
        );
        void refreshServices();
      }
      return;
    }
    window.setTimeout(
      () =>
        updateService(id, {
          state: "running",
          gpu: "GPU 0 · 1 卡",
          memory: "18.7 / 24 GB",
        }),
      900,
    );
  }

  async function stopService(id: string) {
    setNotice("正在优雅停止服务，完成中的请求结束后释放 GPU。");
    updateService(id, { state: "stopping", exposure: "offline" });
    if (backendMode) {
      try {
        const stopped = await requestServiceAction<InferenceService>(id, "/stop", {
          method: "POST",
        });
        updateService(id, normalizeService(stopped));
        setNotice("服务已停止，GPU 资源已释放。");
      } catch (error) {
        setNotice(
          error instanceof Error ? error.message : "停止推理服务失败。",
        );
        void refreshServices();
      }
      return;
    }
    window.setTimeout(
      () =>
        updateService(id, {
          state: "stopped",
          gpu: "未占用",
          memory: "0 / 24 GB",
        }),
      800,
    );
  }

  async function restartService(id: string) {
    setNotice("正在重启服务，接口会短暂不可用。");
    updateService(id, { state: "stopping", exposure: "offline" });
    if (backendMode) {
      try {
        const restarted = await requestServiceAction<InferenceService>(id, "/restart", {
          method: "POST",
        });
        updateService(id, normalizeService(restarted));
        setNotice("服务已重启并通过健康检查。");
      } catch (error) {
        setNotice(
          error instanceof Error ? error.message : "重启推理服务失败。",
        );
        void refreshServices();
      }
      return;
    }
    window.setTimeout(
      () =>
        updateService(id, {
          state: "starting",
          gpu: "GPU 0 · 分配中",
          memory: "加载中",
        }),
      650,
    );
    window.setTimeout(
      () =>
        updateService(id, {
          state: "running",
          exposure: "test",
          gpu: "GPU 0 · 1 卡",
          memory: "18.7 / 24 GB",
        }),
      1450,
    );
  }

  async function loadRetrySpecs(gpuType: string, current?: { cpu: string; memory: string }, serviceIdOverride?: string) {
    if (!backendMode) return;
    setRetryPicker((value) => value ? { ...value, gpuType, loading: true, error: "" } : value);
    try {
      const serviceId = serviceIdOverride ?? retryPicker?.serviceId;
      const service = serviceId ? services.find((item) => item.id === serviceId) : undefined;
      const savedResource = service?.config?.youyunResource;
      const imageId = savedResource?.imageId;
      const query = new URLSearchParams({ gpuType });
      if (imageId) query.set("imageId", String(imageId));
      if (savedResource?.region) query.set("region", String(savedResource.region));
      if (savedResource?.zone) query.set("zone", String(savedResource.zone));
      if (savedResource?.instanceMode) query.set("instanceMode", String(savedResource.instanceMode));
      const specs = await inferenceRequest<YouyunSpecs>(`/youyun/gpu-specs?${query.toString()}`);
      const available = specs.capacity.filter((item) => item.gpu === 1 && item.available);
      const preferred = available.find(
        (item) => item.cpu === Number(current?.cpu) && item.memoryGb === Number(current?.memory),
      ) ?? available[0];
      setRetryPicker((value) => value ? {
        ...value,
        gpuType: specs.gpuType,
        cpu: preferred ? String(preferred.cpu) : "",
        memory: preferred ? String(preferred.memoryGb) : "",
        specs,
        loading: false,
        error: "",
      } : value);
    } catch (error) {
      setRetryPicker((value) => value ? {
        ...value,
        specs: null,
        loading: false,
        error: error instanceof Error ? error.message : "优云 GPU 库存查询失败。",
      } : value);
    }
  }

  async function recoverService(id: string, youyunResource?: Record<string, unknown>) {
    const target = services.find((service) => service.id === id);
    const conflict =
      target && serviceProvider(target) === "local"
        ? services.find(
            (service) =>
              service.id !== id &&
              serviceProvider(service) === "local" &&
              activeStates.includes(service.state),
          )
        : undefined;
    if (conflict) {
      setNotice("本地重试需要占用本机 GPU，请先停止“" + conflict.name + "”。");
      return;
    }
    setNotice("正在按原配置重试，完成前不可重复点击。");
    updateService(id, {
      state: "recovering",
      error: undefined,
      gpu: "GPU 0 · 重试中",
      memory: "加载中",
    });
    if (backendMode) {
      try {
        const recovered = await requestServiceAction<InferenceService>(id, "/recover", {
          method: "POST",
          headers: youyunResource ? { "Content-Type": "application/json" } : undefined,
          body: youyunResource ? JSON.stringify({ youyunResource }) : undefined,
        });
        updateService(id, normalizeService(recovered));
        setNotice("服务已重试并通过健康检查。");
      } catch (error) {
        const message = error instanceof Error ? error.message : "重试推理服务失败。";
        if (!youyunResource && message.includes("库存")) {
          const service = services.find((item) => item.id === id);
          const resource = service?.config?.youyunResource;
          if (service && resource) {
            setRetryPicker({
              serviceId: id,
              gpuType: String(resource.gpuType ?? ""),
              cpu: String(resource.cpuCores ?? ""),
              memory: String(resource.memoryGb ?? ""),
              specs: null,
              loading: true,
              error: "",
            });
            void loadRetrySpecs(String(resource.gpuType ?? ""), {
              cpu: String(resource.cpuCores ?? ""),
              memory: String(resource.memoryGb ?? ""),
            }, id);
            setNotice("原优云 GPU 暂无库存，请重新选择 GPU。");
          } else {
            setNotice(message);
          }
        } else {
          setNotice(message);
        }
        void refreshServices();
      }
      return;
    }
    window.setTimeout(
      () =>
        updateService(id, {
          state: "running",
          exposure: "test",
          gpu: "GPU 0 · 1 卡",
          memory: "18.7 / 24 GB",
        }),
      1100,
    );
  }

  async function submitRetryWithResource() {
    if (!retryPicker || retryPicker.loading || !retryPicker.gpuType || !retryPicker.cpu || !retryPicker.memory) return;
    const service = services.find((item) => item.id === retryPicker.serviceId);
    if (!service) return;
    const previous = service.config?.youyunResource ?? {};
    setRetryPicker(null);
    await recoverService(retryPicker.serviceId, {
      ...previous,
      mode: "manual",
      gpuType: retryPicker.gpuType,
      gpuCount: 1,
      cpuCores: Number(retryPicker.cpu),
      memoryGb: Number(retryPicker.memory),
    });
  }

  function requestArchive(id: string) {
    const service = services.find((item) => item.id === id);
    if (!service) return;
    if (
      ["starting", "running", "stopping", "recovering"].includes(service.state)
    ) {
      setNotice("运行中的服务不能归档，请先停止服务。");
      return;
    }
    setConfirmAction({ type: "archive", serviceId: id });
  }

  function requestDelete(id: string) {
    setConfirmAction({ type: "delete", serviceId: id });
  }

  async function restoreService(id: string) {
    if (backendMode) {
      try {
        const restored = await requestServiceAction<InferenceService>(id, "/restore", {
          method: "POST",
        });
        updateService(id, normalizeService(restored));
      } catch (error) {
        setNotice(
          error instanceof Error ? error.message : "恢复归档服务失败。",
        );
        void refreshServices();
        return;
      }
    }
    updateService(id, {
      state: "stopped",
      exposure: "offline",
      gpu: "未占用",
      memory: "0 / 24 GB",
      error: undefined,
    });
    setNotice("服务已恢复为已停止状态，不会自动启动，也不会占用 GPU。");
  }

  async function confirmServiceAction() {
    if (!confirmAction) return;
    const { type, serviceId } = confirmAction;
    setConfirmAction(null);
    if (type === "archive") {
      if (backendMode) {
        try {
          const archived = await requestServiceAction<InferenceService>(serviceId, "/archive", {
            method: "POST",
          });
          updateService(serviceId, normalizeService(archived));
        } catch (error) {
          setNotice(
            error instanceof Error ? error.message : "归档推理服务失败。",
          );
          void refreshServices();
          return;
        }
      }
      updateService(serviceId, {
        state: "archived",
        exposure: "offline",
        gpu: "未占用",
        memory: "0 / 24 GB",
      });
      setNotice("服务已归档。配置、日志、指标和训练产物关联仍会保留。");
      if (backendMode) void refreshServices();
      if (selectedId === serviceId) setView("list");
      return;
    }
    if (backendMode) {
      try {
        await inferenceRequest<void>(`/inference/services/${serviceId}`, {
          method: "DELETE",
        });
      } catch (error) {
        setNotice(
          error instanceof Error ? error.message : "删除推理服务失败。",
        );
        void refreshServices();
        return;
      }
    }
    setServices((current) => {
      const remaining = current.filter((service) => service.id !== serviceId);
      setSelectedId(remaining[0]?.id ?? "");
      return remaining;
    });
    setView("list");
    setNotice("服务已永久删除；训练任务和 Adapter 不受影响。");
  }

  async function toggleExposure(service: InferenceService) {
    if (service.state !== "running") {
      setNotice(
        "只有运行中的服务可以上线；上线仅代表允许小组内部系统调用，不暴露公网。",
      );
      return;
    }
    const exposure = service.exposure === "online" ? "offline" : "online";
    if (backendMode) {
      try {
        const updated = await requestServiceAction<InferenceService>(service.id, "/exposure", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ exposure }),
        });
        updateService(service.id, normalizeService(updated));
      } catch (error) {
        setNotice(
          error instanceof Error ? error.message : "更新上线状态失败。",
        );
        void refreshServices();
        return;
      }
    }
    updateService(service.id, { exposure });
    setNotice(
      service.exposure === "online"
        ? "服务已下线，运行进程仍保留。"
        : "服务已上线，可供未来内部模块正式调用。",
    );
  }

  async function validateCustomCommand() {
    if (!selectedArtifact) {
      setNotice("当前没有可部署的 Adapter，请先完成训练任务。");
      return false;
    }
    if (!form.customCommand.trim()) {
      setNotice("请粘贴 vLLM 启动命令。");
      setCommandPreview(null);
      return false;
    }
    if (!backendMode) {
      setNotice("连接推理服务后端后，平台才能解析并校验自定义命令。");
      return false;
    }
    setCommandValidating(true);
    try {
      const preview = await inferenceRequest<{
        command: string;
        flags: string[];
        note?: string;
      }>("/inference/commands/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          taskId: selectedArtifact.taskId,
          adapterVariant: selectedArtifact.variant,
          command: form.customCommand.trim(),
        }),
      });
      setCommandPreview(preview);
      setNotice(
        "命令校验通过。平台会在启动时接管服务 ID、端口、Base Model 和所选 Adapter。",
      );
      return true;
    } catch (error) {
      setCommandPreview(null);
      setNotice(
        error instanceof Error ? error.message : "自定义 vLLM 命令校验失败。",
      );
      return false;
    } finally {
      setCommandValidating(false);
    }
  }

  async function createService() {
    if (!form.name.trim()) {
      setNotice("请填写服务名称。");
      return;
    }
    if (!selectedArtifact) {
      setNotice("当前没有可部署的 Adapter，请先完成训练任务。");
      return;
    }
    if (backendMode && modelContextLoading) {
      setNotice("正在读取 Base Model 的上下文能力，请稍候再创建服务。");
      return;
    }
    if (form.deploymentMode === "custom" && !form.customCommand.trim()) {
      setNotice("请选择自定义命令行后粘贴 vLLM 启动命令。");
      return;
    }
    if (backendMode && form.provider === "youyun") {
      if (youyunError) {
        setNotice(`优云 GPU 不可用：${youyunError}`);
        return;
      }
      if (!selectedYouyunImageId) {
        setNotice("请选择优云官方 vLLM 镜像后再创建服务。");
        return;
      }
      if (youyunLoading || !youyunSpecs || !youyunResource) {
        setNotice(youyunLoading ? "正在查询当前地域的 GPU 与 CPU / 内存库存，请稍后再提交。" : youyunPriceError || youyunError || (youyunPriceLoading ? "正在查询优云规格价格，请稍后再提交。" : "请先选择有库存的优云 GPU 规格。"));
        return;
      }
    }
    if (backendMode) {
      setNotice("正在保存推理服务配置，不会立即占用 GPU。");
      try {
        if (form.deploymentMode === "custom") {
          const valid = await validateCustomCommand();
          if (!valid) return;
        }
        const created = await inferenceRequest<InferenceService>(
          "/inference/services",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: form.name.trim(),
              description: form.description.trim(),
              taskId: selectedArtifact.taskId,
              adapterVariant: selectedArtifact.variant,
              owner: "平台管理员",
              provider: form.provider,
              youyunResource:
                form.provider === "youyun"
                  ? {
                      mode: "manual",
                      purpose: "inference",
                      gpuType: youyunResource?.gpuType ?? form.youyunGpuType,
                      gpuCount: 1,
                      cpuCores: Number(form.youyunCpu),
                      memoryGb: Number(form.youyunMemory),
                      imageId: youyunResource?.imageId ?? selectedYouyunImageId,
                      imageName: youyunResource?.imageName ?? selectedYouyunImage?.name ?? selectedYouyunImageId,
                       imageVersion: youyunResource?.imageVersion ?? youyunInferenceImageVersion(selectedYouyunImage),
                      region: selectedYouyunRegion || youyunResource?.region,
                      zone: selectedYouyunZone || youyunResource?.zone,
                      instanceMode: selectedYouyunMode,
                    }
                  : null,
              maxModelLength: Number(form.maxLength),
              dtype: form.dtype,
              quantization: form.quantization,
              gpuMemoryUtilization: Number(form.gpuMemory),
              kvCacheMode: "auto",
              kvCacheMemoryGiB: null,
              kvCacheDtype: "auto",
              maxNumSeqs: Number(form.maxSequences),
              maxNumBatchedTokens: Number(form.batchTokens),
              enablePrefixCaching: form.prefixCaching,
              enableChunkedPrefill: form.chunkedPrefill,
              tensorParallelSize: 1,
              loadTimeout: Number(form.timeout),
              deploymentMode: form.deploymentMode,
              customCommand:
                form.deploymentMode === "custom"
                  ? form.customCommand.trim()
                  : null,
            }),
          },
        );
        setServices((current) => [
          created,
          ...current.filter((service) => service.id !== created.id),
        ]);
        setSelectedId(created.id);
        clearSessionDraft(INFERENCE_DRAFT_STORAGE_KEY);
        setView("detail");
        setTab("overview");
        setNotice("服务配置已创建。创建配置不占 GPU，可在详情页手动启动。");
      } catch (error) {
        setNotice(
          error instanceof Error ? error.message : "创建推理服务失败。",
        );
      }
      return;
    }
    const id = "svc-260818-" + String(services.length + 1).padStart(3, "0");
    const service: InferenceService = {
      id,
      name: form.name.trim(),
      description: form.description.trim() || "未填写服务说明",
      state: "created",
      exposure: "test",
      model: selectedArtifact.model.replace("Qwen/", ""),
      adapter: `${selectedArtifact.variant === "best" ? "Best" : "Final"} Adapter · Step ${selectedArtifact.step}`,
      taskId: selectedArtifact.taskId,
      config: { adapterVariant: selectedArtifact.variant },
      gpu: "未占用",
      memory: "0 / 24 GB",
      endpoint: "/v1/chat/completions",
      createdAt: "2026-08-18 23:20",
      updatedAt: "刚刚",
      recoveryAttempts: 0,
    };
    setServices((current) => [service, ...current]);
    setSelectedId(id);
    clearSessionDraft(INFERENCE_DRAFT_STORAGE_KEY);
    setView("detail");
    setTab("overview");
    setNotice("服务配置已创建。创建配置不占 GPU，可在详情页手动启动。");
  }

  async function sendMessage() {
    const content = prompt.trim();
    if (
      !content ||
      sending ||
      !selectedService ||
      selectedService.state !== "running"
    )
      return;
    const userId = `${Date.now()}-user`;
    const assistantId = `${Date.now()}-assistant`;
    const userMessage: ChatMessage = {
      id: userId,
      role: "user",
      content,
      meta: "",
    };
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      meta: chatParameters.stream ? "生成中 · SSE" : "生成中",
    };
    const conversation = [...messages, userMessage];
    setMessages((current) => [...current, userMessage, assistantMessage]);
    setPrompt("");
    setSending(true);
    const startedAt = performance.now();
    const updateAssistant = (patch: Partial<ChatMessage>) => {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId ? { ...message, ...patch } : message,
        ),
      );
    };

    try {
      if (!backendMode) {
        const answer =
          "当前为本地原型模式。连接推理后端后，这里会通过 vLLM + 所选 Adapter 返回真实回答。";
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        updateAssistant({
          content: answer,
          meta: `${((performance.now() - startedAt) / 1000).toFixed(1)} 秒 · 原型流式输出`,
        });
        return;
      }

      const controller = new AbortController();
      chatAbortRef.current = controller;
      const response = await fetch(
        apiUrl(`/inference/services/${selectedService.id}/chat`),
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: conversation.map((message) => ({
              role: message.role,
              content: message.content,
            })),
            maxTokens: Math.min(chatParameters.maxTokens, chatOutputLimit),
            temperature: chatParameters.temperature,
            topP: chatParameters.topP,
            stream: chatParameters.stream,
          }),
          signal: controller.signal,
        },
      );
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(
          formatApiError(body, `在线调试请求失败（${response.status}）`),
        );
      }
      if (!chatParameters.stream) {
        const body = (await response.json()) as {
          choices?: Array<{
            message?: { content?: string; reasoning_content?: string };
          }>;
          usage?: { completion_tokens?: number };
        };
        const answer =
          body.choices?.[0]?.message?.content ??
          body.choices?.[0]?.message?.reasoning_content ??
          "模型未返回可见文本。";
        updateAssistant({
          content: answer,
          meta: `${((performance.now() - startedAt) / 1000).toFixed(1)} 秒 · ${body.usage?.completion_tokens ?? "—"} tokens · 真实 vLLM JSON`,
        });
        return;
      }
      if (!response.body) throw new Error("推理服务没有返回流式响应。");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";
      let completionTokens: number | undefined;
      const consume = (block: string) => {
        for (const line of block.split(/\r?\n/)) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw || raw === "[DONE]") continue;
          try {
            const chunk = JSON.parse(raw) as {
              choices?: Array<{
                delta?: { content?: string; reasoning_content?: string };
              }>;
              usage?: { completion_tokens?: number };
            };
            const delta = chunk.choices?.[0]?.delta;
            const next = delta?.content ?? delta?.reasoning_content ?? "";
            if (next) {
              answer += next;
              updateAssistant({ content: answer });
            }
            if (chunk.usage?.completion_tokens)
              completionTokens = chunk.usage.completion_tokens;
          } catch {
            // Ignore incomplete/non-JSON SSE lines and continue reading the stream.
          }
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE permits LF or CRLF line endings; normalize the event boundary
        // so streamed tokens are rendered as soon as each event arrives.
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() ?? "";
        blocks.forEach(consume);
      }
      buffer += decoder.decode();
      if (buffer.trim()) consume(buffer);
      updateAssistant({
        content: answer || "模型未返回可见文本。",
        meta: `${((performance.now() - startedAt) / 1000).toFixed(1)} 秒 · ${completionTokens ?? "—"} tokens · 真实 vLLM SSE`,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      const message =
        error instanceof Error ? error.message : "在线调试请求失败。";
      updateAssistant({ content: message, meta: "请求失败" });
      setNotice(message);
    } finally {
      chatAbortRef.current = null;
      setSending(false);
    }
  }

  function serviceApiPath(service: InferenceService) {
    return `/inference/services/${service.id}/chat`;
  }

  function serviceApiEndpoint(service: InferenceService) {
    return apiUrl(serviceApiPath(service));
  }

  function serviceCurlCommand(service: InferenceService) {
    return [
      "curl " + serviceApiEndpoint(service) + " \\",
      "  -H \"Content-Type: application/json\" \\",
      "  -d '{",
      '    "messages": [{"role": "user", "content": "请介绍退款流程"}],',
      '    "maxTokens": 512,',
      '    "temperature": 0.7,',
      '    "topP": 0.9,',
      '    "stream": true',
      "  }'",
    ].join("\n");
  }

  function copyEndpoint(service: InferenceService) {
    void navigator.clipboard?.writeText(serviceApiEndpoint(service));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  function statusBadge(service: InferenceService) {
    const meta = stateMeta[service.state];
    return (
      <span className={"status inference-status " + meta.className}>
        <span className="status-dot" />
        {meta.label}
      </span>
    );
  }

  function exposureBadge(service: InferenceService) {
    const meta = exposureMeta[service.exposure];
    return (
      <span className={"inference-exposure " + meta.className}>
        {meta.label}
      </span>
    );
  }

  function renderList() {
    return (
      <section className="content inference-page">
        <div className="title-row">
          <div>
            <h1>推理服务</h1>
            <div className="subtitle">
              使用 vLLM 部署训练完成的 Base Model + Best
              Adapter，提供在线调试与内部接口
            </div>
          </div>
          <button
            className="primary-btn"
            onClick={() => {
              setNotice("");
              setView("create");
            }}
          >
            <Plus size={15} aria-hidden="true" />创建推理服务
          </button>
        </div>
        <div className="inference-scope-note">
          <span className="inference-note-icon">i</span>
          <div>
            <strong>本地单卡互斥，优云支持多任务并发</strong>
            <p>
              本地训练与本地推理共用一张 GPU；优云训练和多个优云推理服务可同时运行。“上线”仅表示内部正式服务。
            </p>
          </div>
          <span className="inference-future-tag">
            优云按实例独立分配
          </span>
        </div>
        <div
          className={
            "inference-connection-note " +
            (backendMode ? "connected" : "fallback")
          }
        >
          <span className="status-dot" />
          {loadingServices
            ? "正在读取推理服务..."
            : backendMode
              ? storageBackend === "postgres"
                ? "已连接推理服务后端，数据持久化至 PostgreSQL"
                : storageBackend === "sqlite"
                  ? "已连接推理服务后端，当前使用本地 SQLite 兜底"
                  : "已连接推理服务后端，正在确认持久化存储"
              : "当前为原型数据，后端连接后可执行真实服务操作"}
        </div>
        <div className="summary">
          <div className="metric">
            <div className="metric-value">
              {
                services.filter((service) => service.state !== "archived")
                  .length
              }
            </div>
            <div className="metric-label">当前服务</div>
          </div>
          <div className="metric">
            <div className="metric-value blue">
              {services.filter((service) => service.state === "running").length}
            </div>
            <div className="metric-label">运行中</div>
          </div>
          <div className="metric">
            <div className="metric-value green">
              {
                services.filter((service) => service.exposure === "online")
                  .length
              }
            </div>
            <div className="metric-label">已上线</div>
          </div>
          <div className="metric">
            <div className="metric-value red">
              {services.filter((service) => service.state === "failed").length}
            </div>
            <div className="metric-label">启动失败</div>
          </div>
        </div>
        <div className="filters">
          <input
            className="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索服务、模型或训练任务"
            aria-label="搜索服务、模型或训练任务"
          />
          <select
            className="filter-select"
            value={stateFilter}
            onChange={(event) => setStateFilter(event.target.value)}
            aria-label="服务状态筛选"
          >
            <option value="all">全部状态</option>
            {Object.keys(stateMeta).map((state) => (
              <option key={state} value={state}>
                {stateMeta[state as ServiceState].label}
              </option>
            ))}
          </select>
          <div className="filter-spacer" />
          <button
            className="secondary-btn refresh"
            title="刷新列表"
            aria-label="刷新列表"
            type="button"
            onClick={() => void refreshServices()}
            disabled={loadingServices}
          >
            <RefreshCw size={15} aria-hidden="true" />
          </button>
        </div>
        {notice && (
          <div className="inference-inline-notice">
            {notice}
            <button onClick={() => setNotice("")}>×</button>
          </div>
        )}
        <div className="table-wrap">
          <div className="table-scroll">
            <table className="inference-table">
              <thead>
                <tr>
                  <th>服务名称</th>
                  <th>运行状态</th>
                  <th>上线状态</th>
                  <th>模型与 Adapter</th>
                  <th>GPU 资源</th>
                  <th>创建时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {filteredServices.map((service) => (
                  <tr key={service.id}>
                    <td>
                      <button
                        className="dataset-link"
                        onClick={() => openDetail(service.id)}
                      >
                        {service.name}
                      </button>
                      <div className="row-sub">
                        {service.id} · {service.description}
                      </div>
                    </td>
                    <td>
                      {statusBadge(service)}
                      <div className="row-sub">
                        {stateMeta[service.state].hint}
                      </div>
                    </td>
                    <td>{exposureBadge(service)}</td>
                    <td>
                      <strong className="inference-model-name">
                        {service.model}
                      </strong>
                      <div className="row-sub">{service.adapter}</div>
                    </td>
                    <td>
                      {service.gpu}
                      <div className="row-sub">显存 {service.memory}</div>
                    </td>
                    <td>
                      {formatServiceCreatedAt(service.createdAt)}
                    </td>
                    <td>
                      <div className="table-actions">
                        <button onClick={() => openDetail(service.id)}>
                          {service.state === "archived" ? "查看" : "详细"}
                        </button>
                        {service.state === "archived" ? (
                          <>
                            <button disabled={!localRuntimeEnabled && serviceProvider(service) === "local"} title={!localRuntimeEnabled && serviceProvider(service) === "local" ? LOCAL_RUNTIME_DISABLED : undefined}
                              onClick={() => void restoreService(service.id)}
                            >
                              恢复
                            </button>
                            <button
                              className="danger-text"
                              onClick={() => requestDelete(service.id)}
                            >
                              删除
                            </button>
                          </>
                        ) : service.state === "running" ? (
                          <>
                            <button
                              onClick={() => openDetail(service.id, "debug")}
                            >
                              调试
                            </button>
                            <button
                              className="danger-text"
                              onClick={() => void stopService(service.id)}
                            >
                              停止
                            </button>
                          </>
                        ) : service.state === "failed" ? (
                          <>
                            <button disabled={!localRuntimeEnabled && serviceProvider(service) === "local"} title={!localRuntimeEnabled && serviceProvider(service) === "local" ? LOCAL_RUNTIME_DISABLED : undefined}
                              onClick={() => void recoverService(service.id)}
                            >
                              重试
                            </button>
                            <button onClick={() => requestArchive(service.id)}>
                              归档
                            </button>
                          </>
                        ) : service.state === "starting" ? (
                          <button
                            className="danger-text"
                            onClick={() => void stopService(service.id)}
                          >
                            停止
                          </button>
                        ) : service.state === "recovering" ? (
                          <>
                            <button className="retrying-action" disabled>重试中</button>
                            <button className="danger-text" onClick={() => void stopService(service.id)}>停止</button>
                          </>
                        ) : service.state === "stopping" ? null : (
                          <>
                            <button disabled={!localRuntimeEnabled && serviceProvider(service) === "local"} title={!localRuntimeEnabled && serviceProvider(service) === "local" ? LOCAL_RUNTIME_DISABLED : undefined}
                              onClick={() => void startService(service.id)}
                            >
                              启动
                            </button>
                            <button onClick={() => requestArchive(service.id)}>
                              归档
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        {!filteredServices.length && (
            <div className="empty">没有符合筛选条件的推理服务。</div>
          )}
          <Pagination page={servicePage} totalPages={servicePageInfo.totalPages} total={servicePageInfo.total} unit="条服务" onPageChange={setServicePage} />
        </div>
      </section>
    );
  }

  function field(label: React.ReactNode, control: React.ReactNode) {
    return (
      <div className="inference-field">
        <label>{label}</label>
        {control}
      </div>
    );
  }

  function renderCreate() {
    if (!selectedArtifact) {
      return (
        <section className="content inference-page">
          <div className="title-row">
            <div>
              <BackButton label="返回推理服务" onClick={leaveCreate} />
              <h1>创建推理服务</h1>
              <div className="subtitle">
                服务创建后默认不启动，不会立即占用 GPU
              </div>
            </div>
          </div>
          <div className="inference-empty-artifact">
            <strong>暂无可部署的训练产物</strong>
            <p>
              已完成或已停止的训练任务只要存在真实 Best 或 Final Adapter，
              就可以创建推理服务。
            </p>
            <button
              className="secondary-btn"
              onClick={() => void refreshServices()}
            >
              重新读取训练产物
            </button>
          </div>
        </section>
      );
    }
    return (
      <section className="content inference-page">
        <div className="title-row">
          <div>
            <BackButton label="返回推理服务" onClick={leaveCreate} />
            <h1>创建推理服务</h1>
            <div className="subtitle">
              服务创建后默认不启动，不会立即占用 GPU
            </div>
          </div>
        </div>
        <div className="inference-steps">
          <div className="active">
            <span>1</span>选择训练产物
          </div>
          <i />
          <div className="active">
            <span>2</span>配置 vLLM
          </div>
          <i />
          <div>
            <span>3</span>确认并创建
          </div>
        </div>
        {notice && (
          <div className="inference-inline-notice warning-notice">
            {notice}
            <button onClick={() => setNotice("")}>×</button>
          </div>
        )}
        <div
          className={
            "inference-create-layout " +
            (form.deploymentMode === "custom" ? "inference-custom-layout" : "")
          }
        >
          <div
            className={
              "inference-form-stack " +
              (form.deploymentMode === "custom" ? "inference-custom-mode" : "")
            }
          >
            <section className="panel inference-form-section">
              <div className="section-heading">
                <div>
                  <h2>选择训练产物</h2>
                  <p>
                    可从已完成或已停止任务中选择 Best 或 Final Adapter，并加载对应 Base Model。
                  </p>
                </div>
                <span className="section-index">01</span>
              </div>
              {field(
                "训练任务",
                <select
                  value={form.artifactId}
                  onChange={(event) =>
                    setForm({ ...form, artifactId: event.target.value })
                  }
                >
                  {availableArtifacts.map((artifact) => (
                    <option
                      key={artifact.id}
                      value={artifact.id}
                      disabled={!artifact.deployable}
                    >
                      {artifact.name} · {artifact.taskId} · {artifact.variant === "best" ? "Best" : "Final"}
                      {artifact.deployable ? "" : "（不可部署）"}
                    </option>
                  ))}
                </select>,
              )}
              <div className="inference-artifact-lock">
                <div>
                  <span>Base Model</span>
                  <strong>{selectedArtifact.model}</strong>
                  <small>从训练任务读取，不允许替换</small>
                </div>
                <b>＋</b>
                <div>
                  <span>{selectedArtifact.variant === "best" ? "Best" : "Final"} Adapter</span>
                  <strong>{selectedArtifact.adapter}</strong>
                  <small>
                    {selectedArtifact.evalLoss == null
                      ? "无 Eval Loss"
                      : `Eval Loss ${selectedArtifact.evalLoss.toFixed(4)}`} · 用户选择
                  </small>
                </div>
                <span className="inference-lock">已锁定</span>
              </div>
              {backendMode && !modelContext && (
                <div className="inference-capability-status">
                  {modelContextLoading
                    ? "正在读取所选 Base Model 的参数能力，读取期间暂时锁定上下文和输出上限。"
                    : "暂时无法读取 Base Model 上限；当前值仍可调整，提交时服务端会再次校验。"}
                </div>
              )}
              <div className="inference-rule">
                <strong>部署边界</strong>
                <span>
                  不支持仅部署 Base Model；可按训练产物选择 Best 或 Final
                  Adapter。合并模型不在本次 MVP 内。
                </span>
              </div>
            </section>
            <section className="panel inference-form-section">
              <div className="section-heading">
                <div>
                  <h2>基础配置</h2>
                  <p>优先展示小白常用参数，默认值可直接创建。</p>
                </div>
                <span className="section-index">02</span>
              </div>
              <div className="inference-deployment-mode">
                <div className="inference-subsection-heading">
                  <div>
                    <strong>部署方式</strong>
                    <p>
                      标准配置适合大多数情况；命令行模式用于粘贴复杂的 vLLM
                      启动参数。
                    </p>
                  </div>
                  <span className="inference-mode-hint">创建时不占 GPU</span>
                </div>
                <div className="inference-mode-options">
                  <button
                    type="button"
                    className={
                      form.deploymentMode === "managed" ? "selected" : ""
                    }
                    onClick={() => {
                      setForm({
                        ...form,
                        deploymentMode: "managed",
                        customCommand: "",
                      });
                      setCommandPreview(null);
                    }}
                  >
                    <span className="inference-mode-radio" />
                    <span>
                      <strong>标准配置</strong>
                      <small>
                        由平台生成安全的 vLLM 启动命令，适合小白快速部署。
                      </small>
                    </span>
                  </button>
                  {localRuntimeEnabled && <button
                    type="button"
                    className={
                      form.deploymentMode === "custom" ? "selected" : ""
                    }
                    onClick={() =>
                      setForm({ ...form, deploymentMode: "custom" })
                    }
                  >
                    <span className="inference-mode-radio" />
                    <span>
                      <strong>自定义命令行</strong>
                      <small>
                        粘贴复杂的 vLLM 命令，平台校验后保存，启动时才执行。
                      </small>
                    </span>
                  </button>}
                </div>
                {form.deploymentMode === "custom" && (
                  <div className="inference-custom-command-panel">
                    <div className="inference-custom-command-head">
                      <div>
                        <strong>粘贴 vLLM 启动命令</strong>
                        <p>
                          仅支持 vLLM 服务命令，不支持任意 Shell
                          脚本、管道、重定向或 bash -c。
                        </p>
                      </div>
                      <button
                        type="button"
                        className="secondary-btn compact-btn"
                        onClick={() => void validateCustomCommand()}
                        disabled={
                          commandValidating || !form.customCommand.trim()
                        }
                      >
                        {commandValidating ? "校验中…" : "解析并校验"}
                      </button>
                    </div>
                    <textarea
                      className="inference-command-editor"
                      rows={8}
                      value={form.customCommand}
                      onChange={(event) => {
                        setForm({ ...form, customCommand: event.target.value });
                        setCommandPreview(null);
                      }}
                      placeholder={
                        "例如：\nvllm serve Qwen/Qwen3-1.7B \\\n  --lora-modules adapter=/path/to/best-adapter \\\n  --max-model-len 4096 \\\n  --gpu-memory-utilization 0.90"
                      }
                      spellCheck={false}
                    />
                    <div className="inference-command-help">
                      <span>可复制粘贴 vLLM 命令</span>
                      <span>
                        平台仍会强制绑定已选 Base Model + 所选 Adapter
                      </span>
                    </div>
                    {commandPreview && (
                      <div className="inference-command-preview">
                        <div className="inference-preview-title">
                          <strong>解析结果</strong>
                          <span>校验通过</span>
                        </div>
                        <div className="inference-preview-grid">
                          <div>
                            <span>Base Model</span>
                            <strong>
                              {commandPreview.model ?? selectedArtifact.model}
                            </strong>
                          </div>
                          <div>
                            <span>{selectedArtifact.variant === "best" ? "Best" : "Final"} Adapter</span>
                            <strong>
                              {commandPreview.adapter ??
                                selectedArtifact.adapter}
                            </strong>
                          </div>
                          <div>
                            <span>端口</span>
                            <strong>{commandPreview.port ?? "平台分配"}</strong>
                          </div>
                          <div>
                            <span>自定义参数</span>
                            <strong>
                              {commandPreview.flags.length
                                ? `${commandPreview.flags.length} 项`
                                : "无"}
                            </strong>
                          </div>
                        </div>
                        {commandPreview.flags.length > 0 && (
                          <div className="inference-command-flags">
                            <span>识别到的参数</span>
                            <code>{commandPreview.flags.join("  ")}</code>
                          </div>
                        )}
                        <pre className="inference-command-normalized">
                          {commandPreview.command}
                        </pre>
                        <small>
                          {commandPreview.note ??
                            "启动时平台会接管服务 ID、端口、Base Model 和所选 Adapter。"}
                        </small>
                      </div>
                    )}
                  </div>
                )}
              </div>
              <div className="inference-form-grid two-columns">
                {field(
                  <span>
                    服务名称 <b className="inference-required-mark">*</b>
                  </span>,
                  <input
                    required
                    aria-required="true"
                    value={form.name}
                    onChange={(event) =>
                      setForm({ ...form, name: event.target.value })
                    }
                    placeholder="例如：客服问答-测试服务"
                  />,
                )}
                {field(
                  "推理引擎",
                  <>
                    <input value="vLLM" disabled />
                    <small>当前部署 MVP 固定使用 vLLM</small>
                  </>,
                )}
                {field(
                  "运行环境",
                  <>
                    <div className="inference-provider-options">
                      {localRuntimeEnabled && <button
                        type="button"
                        className={form.provider === "local" ? "selected" : ""}
                        onClick={() => setForm({ ...form, provider: "local" })}
                      >
                        <span className="inference-mode-radio" />
                        <span>
                          <strong>本地 GPU / CPU</strong>
                          <small>沿用当前本地 vLLM 运行能力</small>
                        </span>
                      </button>}
                      <button
                        type="button"
                        className={form.provider === "youyun" ? "selected" : ""}
                        onClick={() => setForm({ ...form, provider: "youyun" })}
                      >
                        <span className="inference-mode-radio" />
                        <span>
                          <strong>优云远程 GPU</strong>
                          <small>一次性实例 · 按量计费</small>
                        </span>
                      </button>
                    </div>
                    <small>
                      {form.provider === "youyun"
                        ? "实时库存与价格会在创建前校验，公网访问暂不开放"
                        : "沿用当前本地 vLLM 运行能力"}
                    </small>
                  </>,
                )}
                {form.provider === "youyun" && (
                  <div className="full">
                    <div className="youyun-specs-panel">
                      <div className="youyun-specs-head">
                        <div>
                          <strong>优云 GPU 配置与实时库存</strong>
                          <small>
                            {youyunSpecs
                              ? `${youyunRegionLabel(youyunSpecs.region)} · ${youyunSpecs.chargeType === "Postpay" ? "按量计费" : youyunSpecs.chargeType}`
                              : "正在查询可用规格..."}
                          </small>
                        </div>
                        <button
                          type="button"
                          className="secondary-btn compact-btn"
                          disabled={youyunLoading}
                           onClick={() => void refreshYouyunSpecs(undefined, undefined, undefined, undefined, undefined, true)}
                        >
                          {youyunLoading ? "查询中..." : "刷新库存"}
                        </button>
                      </div>
                      {youyunError && <div className="youyun-specs-error">{youyunError}</div>}
                      <div className="youyun-image-picker-row">
                        <div>
                          <strong>远程 vLLM 官方镜像</strong>
                          <small>{selectedYouyunImageId ? `镜像：${selectedYouyunImage?.name ?? selectedYouyunImageId} · ID：${selectedYouyunImageId} · ${selectedYouyunImage?.container === true ? "容器镜像" : "虚机镜像"}` : "请选择官方 vLLM 镜像"}</small>
                        </div>
                        <select
                          value={selectedYouyunImageId}
                          disabled={youyunImagesLoading || !youyunImages.length}
                          onChange={(event) => {
                            const imageId = event.target.value;
                            setSelectedYouyunImageId(imageId);
                            setYouyunResource(null);
                             void refreshYouyunSpecs(form.youyunGpuType, imageId, selectedYouyunRegion, selectedYouyunZone, selectedYouyunMode, true);
                          }}
                        >
                          <option value="">{youyunImagesLoading ? "读取官方镜像..." : "选择镜像"}</option>
                          {youyunImages.map((image) => (
                            <option
                              key={image.id}
                              value={image.id}
                              disabled={!getYouyunInferenceImageAdvice(image, form.youyunGpuType).compatible}
                            >
                              {youyunInferenceImageOptionLabel(image, form.youyunGpuType)}
                            </option>
                          ))}
                        </select>
                        <button type="button" className="secondary-btn compact-btn" onClick={() => void refreshYouyunImages()} disabled={youyunImagesLoading}>
                          {youyunImagesLoading ? "查询中..." : "刷新镜像"}
                        </button>
                      </div>
                      {youyunImagesError && <div className="youyun-specs-error">{youyunImagesError}</div>}
                      {selectedYouyunImageAdvice && (
                        <div className={`youyun-image-guidance ${selectedYouyunImageAdvice.tone}`}>
                          <strong>{selectedYouyunImageAdvice.badge}</strong>
                          <span>{selectedYouyunImageAdvice.message}</span>
                          {selectedYouyunImage?.compatibility && (
                            <small>
                              版本 {youyunInferenceImageVersion(selectedYouyunImage) ?? "优云未提供版本信息"} · GPU {selectedYouyunImage.compatibility.supportedGpu?.join("、") || "待确认"}
                            </small>
                          )}
                        </div>
                      )}
                      {youyunSpecs?.zones?.length ? (
                        <div className="youyun-location-picker">
                  <span className="field-label">地域 / 可用区</span>
                  <div className="youyun-region-list">{youyunLocations.map((item) => <button type="button" key={item.zone} disabled={item.supported === false} title={item.disabledReason || item.describe || item.zone} className={`youyun-region-option${(selectedYouyunZone || youyunSpecs.locationZone || youyunSpecs.zone) === item.zone ? " selected" : ""}${item.supported === false ? " disabled" : ""}`} onClick={() => {
                    if (item.supported === false) return;
                    const nextMode = youyunPreferredMode(item, selectedYouyunImage?.container, selectedYouyunMode);
                    setSelectedYouyunRegion(item.region);
                    setSelectedYouyunZone(item.zone);
                    setSelectedYouyunMode(nextMode);
                    setYouyunResource(null);
                    setForm((current) => ({ ...current, youyunGpuType: "", youyunCpu: "", youyunMemory: "" }));
                    setYouyunPriceError("");
                    void refreshYouyunSpecs("", selectedYouyunImageId, item.region, item.zone, nextMode);
                  }}>{item.displayName || item.describe || item.zone}</button>)}</div>
                  <small>每个地域对应优云的精确可用区；GPU 是否可用以当前 Zone 的实时库存为准。</small>
                        </div>
                      ) : null}
                      {youyunSpecs && (
                        <>
                          <div className="youyun-selection-summary">
                            <strong>手动选择单卡 GPU 与 CPU / 内存规格</strong>
                            <small>
                              当前地域：{selectedYouyunLocation?.displayName || youyunRegionLabel(youyunSpecs.region)} · 可用区 {youyunSpecs.zone}；推理服务暂仅支持单卡。价格会随所选规格实时刷新。
                            </small>
                            <small>{youyunGpuHint(form.youyunGpuType, youyunSpecs.gpuMemoryGb)}</small>
                          </div>
                          <div className="youyun-machine-list">
                            {Array.from(
                              new Map(
                                youyunSpecs.machineTypes.map((machine) => [
                                  `${machine.name}-${machine.graphicsMemoryGb}`,
                                  machine,
                                ]),
                              ).values(),
                            ).map((machine) => (
                              <button
                                type="button"
                                key={`${machine.name}-${machine.graphicsMemoryGb}`}
                                className={`youyun-machine-item${youyunMachineIsUnavailable(machine, "inference") ? " sold-out" : ""}${form.youyunGpuType === machine.name ? " selected" : ""}`}
                                disabled={youyunLoading || youyunMachineIsUnavailable(machine, "inference")}
                                onClick={() => {
                                  const recommendedImageId = selectRecommendedYouyunInferenceImageId(
                                    youyunImages,
                                    machine.name,
                                    selectedYouyunImageId,
                                  );
                                  setForm((current) => ({
                                    ...current,
                                    youyunGpuType: machine.name,
                                    youyunCpu: "",
                                    youyunMemory: "",
                                  }));
                                  setSelectedYouyunImageId(recommendedImageId);
                                  setYouyunSpecs((current) => current ? { ...current, gpuType: machine.name, gpuMemoryGb: machine.graphicsMemoryGb, capacity: [] } : current);
                                  setYouyunResource(null);
                                  void refreshYouyunCapacity(machine.name, recommendedImageId, selectedYouyunRegion || youyunSpecs.region, machine.zone || selectedYouyunZone);
                                }}
                              >
                                <strong>{machine.graphicsMemoryGb ? `${machine.graphicsMemoryGb}G` : "GPU"} {machine.name}</strong>
                                <small>{machine.graphicsMemoryGb ? `${machine.graphicsMemoryGb}GB 显存 · ` : ""}{machine.description || machine.zone}{machine.supportReason ? ` · ${machine.supportReason}` : ""}{form.youyunGpuType === machine.name && youyunSpecs.capacity.find((spec) => spec.gpu === 1 && spec.available) ? ` · 默认 ${youyunSpecs.capacity.find((spec) => spec.gpu === 1 && spec.available)!.cpu} 核 / ${youyunSpecs.capacity.find((spec) => spec.gpu === 1 && spec.available)!.memoryGb}GB 内存` : ""}</small>
                                <span className={youyunMachineInventoryClass(machine)}>{youyunMachineInventoryLabel(machine)}</span>
                                <span className={machine.inferenceSupported === false ? "tag-off" : "tag-ready"}>{machine.inferenceSupported === false ? "暂不支持推理" : "支持推理"}</span>
                              </button>
                            ))}
                          </div>
                          <div className="youyun-capacity-list youyun-single-capacity-list">
                            {youyunSpecs.capacity.filter((spec) => spec.gpu === 1).map((spec) => (
                              <button
                                type="button"
                                key={`${spec.gpu}-${spec.cpu}-${spec.memoryGb}`}
                                disabled={youyunLoading || !spec.available}
                                className={`youyun-capacity-item${Number(form.youyunCpu) === spec.cpu && Number(form.youyunMemory) === spec.memoryGb ? " selected" : ""}`}
                                onClick={() => setForm((current) => ({ ...current, youyunCpu: String(spec.cpu), youyunMemory: String(spec.memoryGb) }))}
                              >
                                <span>{spec.cpu} 核 / {spec.memoryGb}GB 内存</span>
                                <span className={spec.available ? "tag-ready" : "tag-off"}>{spec.available ? "可选" : "CPU / 内存无库存"}</span>
                              </button>
                            ))}
                          </div>
                          {youyunSpecs.capacity.some((spec) => spec.gpu !== 1) && (
                            <div className="youyun-multi-capacity-list">
                              <span className="youyun-capacity-label">多卡规格（暂不支持）</span>
                              {youyunSpecs.capacity.filter((spec) => spec.gpu !== 1).map((spec) => (
                                <span className="youyun-capacity-disabled" key={`${spec.gpu}-${spec.cpu}-${spec.memoryGb}`}>{spec.gpu} 卡 / {spec.cpu} 核 / {spec.memoryGb}GB</span>
                              ))}
                            </div>
                          )}
                          <div className="youyun-price-summary">
                            <div>
                              <strong>预计按量费用</strong>
                              <span>
                                {youyunPriceLoading
                                  ? "查询中..."
                                  : youyunResource
                                    ? `${youyunResource.price.currency === "CNY" ? "¥" : `${youyunResource.price.currency} `}${youyunResource.price.total.toFixed(2)} / 小时`
                                    : "暂不可获取"}
                              </span>
                            </div>
                            {youyunResource && (
                              <small>
                                实例 ¥{youyunResource.price.instance.toFixed(2)} · 系统盘 ¥{youyunResource.price.systemDisks.toFixed(2)} · 数据盘 ¥{youyunResource.price.disks.toFixed(2)} · 镜像 ¥{youyunResource.price.image.toFixed(2)}；实际以优云账单为准
                              </small>
                            )}
                          </div>
                          {youyunPriceError && <div className="youyun-specs-error">{youyunPriceError}<button type="button" className="link-btn" onClick={() => void refreshYouyunSpecs(undefined, undefined, undefined, undefined, undefined, true)}>重新查询</button></div>}
                          {youyunLoading && <div role="status">正在更新当前地域的 GPU 与 CPU / 内存库存...</div>}
                          {!youyunLoading && !youyunError && youyunInventoryMessage(youyunSpecs, "inference") && (
                            <div className="youyun-specs-error">{youyunInventoryMessage(youyunSpecs, "inference")}</div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                )}
                {field(
                  "权重量化",
                  <>
                    <select
                      value={form.quantization}
                      onChange={(event) =>
                        setForm({ ...form, quantization: event.target.value })
                      }
                    >
                      <option value="none">不启用权重量化</option>
                      <option value="bitsandbytes">4-bit BitsAndBytes</option>
                    </select>
                    <small>不量化时权重格式由基模与运行时决定；4-bit 可降低 Base Model 显存占用</small>
                  </>,
                )}
                {field(
                  "最大上下文长度",
                  <select
                    value={form.maxLength}
                    disabled={modelContextLoading}
                    onChange={(event) =>
                      setForm({ ...form, maxLength: event.target.value })
                    }
                  >
                    {(contextOptions.length ? contextOptions : [Number(form.maxLength) || 4096]).map((value) => (
                      <option key={value} value={value}>
                        {value.toLocaleString()} tokens
                        {value === contextLimit ? "（基模上限）" : ""}
                      </option>
                    ))}
                  </select>,
                )}
                {field(
                  "服务用途",
                  <textarea
                    className="full"
                    rows={3}
                    value={form.description}
                    onChange={(event) =>
                      setForm({ ...form, description: event.target.value })
                    }
                    placeholder="说明该服务用于测试还是内部正式调用"
                  />,
                )}
              </div>
            </section>
            <section className="panel inference-form-section">
              <button
                className="advanced-toggle inference-advanced-toggle"
                onClick={() => setAdvancedOpen(!advancedOpen)}
              >
                <span>
                  高级 vLLM 配置 <small>已有安全默认值，通常无需修改</small>
                </span>
                <b>{advancedOpen ? "收起 −" : "展开 ＋"}</b>
              </button>
              {advancedOpen && (
                <div className="advanced-panel inference-advanced-panel">
                  <div className="inference-capabilities">
                    <div>
                      <span>PagedAttention</span>
                      <strong>已启用</strong>
                      <small>vLLM 默认能力 · 只读</small>
                    </div>
                    <div>
                      <span>连续批处理</span>
                      <strong>已启用</strong>
                      <small>vLLM 自动调度 · 只读</small>
                    </div>
                    <div>
                      <span>运行引擎</span>
                      <strong>vLLM</strong>
                      <small>当前 MVP 固定</small>
                    </div>
                  </div>
                  <div className="advanced-form-grid">
                    {field(
                      "计算精度 dtype",
                      <select
                        value={form.dtype}
                        onChange={(event) =>
                          setForm({ ...form, dtype: event.target.value })
                        }
                      >
                        <option value="auto">auto（推荐）</option>
                        <option value="float16">float16</option>
                        <option value="bfloat16">bfloat16</option>
                      </select>,
                    )}
                    {field(
                      "GPU 显存利用率",
                      <>
                        <input
                          type="number"
                          min="0.5"
                          max="0.98"
                          step="0.01"
                          value={form.gpuMemory}
                          onChange={(event) =>
                            setForm({ ...form, gpuMemory: event.target.value })
                          }
                        />
                        <small>对应 --gpu-memory-utilization</small>
                      </>,
                    )}
                    {field(
                      "最大并发序列",
                      <input
                        type="number"
                        value={form.maxSequences}
                        onChange={(event) =>
                          setForm({ ...form, maxSequences: event.target.value })
                        }
                      />,
                    )}
                    {field(
                      "批处理 token 上限",
                      <input
                        type="number"
                        value={form.batchTokens}
                        onChange={(event) =>
                          setForm({ ...form, batchTokens: event.target.value })
                        }
                      />,
                    )}
                    {field(
                      "模型加载超时",
                      <>
                        <input
                          type="number"
                          value={form.timeout}
                          onChange={(event) =>
                            setForm({ ...form, timeout: event.target.value })
                          }
                        />
                        <small>秒，超时后进入失败状态</small>
                      </>,
                    )}
                  </div>
                  <div className="inference-readonly-note" role="note">
                    KV Cache dtype、手动容量、Prefix Caching、Chunked Prefill 与 Tensor Parallel 由平台按镜像和单卡资源自动管理。
                  </div>
                </div>
              )}
            </section>
          </div>
          <aside className="panel inference-create-summary">
            <div className="panel-head">
              <div>
                <h2>创建摘要</h2>
                <p>确认部署来源与资源边界</p>
              </div>
              <span className="inference-engine-badge">vLLM</span>
            </div>
            <div className="inference-summary-list">
              <div>
                <span>部署方式</span>
                <strong>
                  {form.deploymentMode === "custom"
                    ? "自定义命令行"
                    : "标准配置"}
                </strong>
              </div>
              <div>
                <span>加载方式</span>
                <strong>Base + {selectedArtifact.variant === "best" ? "Best" : "Final"} Adapter</strong>
              </div>
              <div>
                <span>GPU</span>
                <strong>
                  {form.provider === "youyun"
                    ? `${form.youyunGpuType || "优云 GPU"} · 1 卡`
                    : "1 卡 · 自动选择"}
                </strong>
              </div>
              {form.provider === "youyun" && (
                <>
                  <div>
                    <span>CPU / 内存</span>
                    <strong>{form.youyunCpu || "-"} 核 / {form.youyunMemory || "-"}GB</strong>
                  </div>
                  <div>
                    <span>预计费用</span>
                    <strong>
                      {youyunPriceLoading
                        ? "查询中..."
                        : youyunResource
                          ? `${youyunResource.price.currency === "CNY" ? "¥" : `${youyunResource.price.currency} `}${youyunResource.price.total.toFixed(2)} / 小时`
                          : "暂不可获取"}
                    </strong>
                  </div>
                </>
              )}
              <div>
                <span>运行环境</span>
                <strong>{form.provider === "youyun" ? "优云远程 GPU" : "本地 GPU / CPU"}</strong>
              </div>
              <div>
                <span>Tensor Parallel</span>
                <strong>1</strong>
              </div>
              <div>
                <span>KV Cache</span>
                <strong>
                  {form.kvCacheMode === "auto"
                    ? "启动后自动计算"
                    : form.kvCacheBytes + " GiB / GPU"}
                </strong>
              </div>
              <div>
                <span>接口协议</span>
                <strong>OpenAI Compatible</strong>
              </div>
            </div>
            {form.deploymentMode === "custom" && (
              <div className="inference-custom-summary">
                <strong>命令行状态</strong>
                <span>
                  {commandPreview ? "已校验，可创建" : "请先解析并校验"}
                </span>
              </div>
            )}
            <div className="inference-resource-warning">
              <strong>资源占用提醒</strong>
              <p>
                {form.provider === "youyun"
                  ? "创建配置不占资源。启动时会立即校验所选优云规格；库存不足则直接失败，请更换规格后重试。"
                  : "创建配置不占 GPU。本地启动时若本机 GPU 已被本地训练或推理占用，本服务将拒绝启动。"}
              </p>
            </div>
            <div className="inference-future-box">
              <span>未来配置目标</span>
              <strong>Tensor Parallel 2 / 4 / 8 GPU</strong>
              <p>待资源调度能力落地后开放。</p>
            </div>
            <div className="inference-summary-actions">
              <button className="secondary-btn" onClick={leaveCreate}>
                取消
              </button>
              <button
                className="primary-btn"
                disabled={!form.name.trim()}
                onClick={() => void createService()}
              >
                确认创建
              </button>
            </div>
          </aside>
        </div>
      </section>
    );
  }

  function renderOverview(service: InferenceService) {
    const config = service.config;
    const youyunSummary = youyunResourceSummary(config?.youyunResource);
    const maxModelLength = config?.maxModelLength?.toLocaleString() ?? "4,096";
    const gpuMemoryUtilization =
      config?.gpuMemoryUtilization?.toFixed(2) ?? "0.75";
    const maxNumSeqs = config?.maxNumSeqs?.toLocaleString() ?? "16";
    const maxNumBatchedTokens =
      config?.maxNumBatchedTokens?.toLocaleString() ?? "8,192";
    const prefixCaching = config?.enablePrefixCaching !== false;
    const chunkedPrefill = config?.enableChunkedPrefill !== false;
    return (
      <>
        <>
          {service.error && (
            <div className="inference-error-card">
              <div>
                <strong>失败原因</strong>
                {youyunSummary && <p className="inference-error-resource">本次选择：{youyunSummary}</p>}
                <p>{service.error}</p>
              </div>
            </div>
          )}
        </>
        <div className="inference-detail-grid">
          <section className="panel inference-runtime-card">
            <div className="panel-head">
              <div>
                <h2>运行与资源</h2>
                <p>{youyunSummary ? `远程优云 GPU · ${youyunSummary}` : "当前服务生命周期和 GPU 占用"}</p>
              </div>
              {statusBadge(service)}
            </div>
            <div className="inference-resource-meter">
              <div className="inference-gpu-visual">
                <span>GPU 0</span>
                <strong>{service.state === "running" ? "78%" : "0%"}</strong>
              </div>
              <div>
                <div className="inference-meter-label">
                  <span>显存占用</span>
                  <strong>{service.memory}</strong>
                </div>
                <div className="inference-meter">
                  <span
                    style={{
                      width: service.state === "running" ? "78%" : "0%",
                    }}
                  />
                </div>
                <small>{youyunSummary ? "优云实例独立分配 · 可与其他优云任务并发" : "本地训练与本地推理互斥 · 本机仅运行 1 个 GPU 任务"}</small>
              </div>
            </div>
            <div className="inference-mini-stats">
              <div>
                <span>运行时长</span>
                <strong>
                  {service.state === "running" ? "03:26:18" : "--"}
                </strong>
              </div>
              <div>
                <span>活跃请求</span>
                <strong>{service.state === "running" ? "2" : "0"}</strong>
              </div>
              <div>
                <span>平均首 token</span>
                <strong>{service.state === "running" ? "380 ms" : "--"}</strong>
              </div>
              <div>
                <span>输出吞吐</span>
                <strong>
                  {service.state === "running" ? "31.4 tok/s" : "--"}
                </strong>
              </div>
            </div>
          </section>
          <section className="panel inference-artifact-card">
            <div className="panel-head">
              <div>
                <h2>模型产物</h2>
                <p>部署来源固定且可追溯</p>
              </div>
              <span className="inference-lock">已锁定</span>
            </div>
            <div className="inference-artifact-chain">
              <div>
                <span>Base Model</span>
                <strong>{service.model}</strong>
                <small>训练任务中的基模</small>
              </div>
              <b>＋</b>
              <div>
                <span>{service.config?.adapterVariant === "final" ? "Final" : "Best"} Adapter</span>
                <strong>{service.adapter}</strong>
                <small>{service.config?.adapterVariant === "final" ? "最终训练产物" : "最低验证 Loss 版本"}</small>
              </div>
            </div>
            <button className="inference-task-link">
              训练任务 {service.taskId} ↗
            </button>
          </section>
          <section className="panel inference-config-card">
            <div className="panel-head">
              <div>
                <h2>vLLM 配置</h2>
                <p>创建时保存的运行参数快照</p>
              </div>
              <button
                className="secondary-btn compact-btn"
                onClick={() => setShowConfigDetails(true)}
              >
                查看全部
              </button>
            </div>
            <div className="inference-config-grid">
              <div>
                <span>Engine</span>
                <strong>vLLM</strong>
              </div>
              <div>
                <span>dtype</span>
                <strong>{config?.dtype ?? "auto"}</strong>
              </div>
              <div>
                <span>Max model length</span>
                <strong>{maxModelLength}</strong>
              </div>
              <div>
                <span>GPU memory</span>
                <strong>{gpuMemoryUtilization}</strong>
              </div>
              <div>
                <span>Max sequences</span>
                <strong>{maxNumSeqs}</strong>
              </div>
              <div>
                <span>Tensor parallel</span>
                <strong>{config?.tensorParallelSize ?? 1}</strong>
              </div>
              <div>
                <span>Prefix caching</span>
                <strong>{prefixCaching ? "开启" : "关闭"}</strong>
              </div>
              <div>
                <span>Chunked prefill</span>
                <strong>{chunkedPrefill ? "开启" : "关闭"}</strong>
              </div>
            </div>
            {showConfigDetails && (
              <div
                className="inference-config-modal-backdrop"
                onClick={() => setShowConfigDetails(false)}
              >
                <div
                  ref={configDialogRef}
                  className="inference-config-modal"
                  role="dialog"
                  aria-modal="true"
                  aria-labelledby="inference-config-title"
                  tabIndex={-1}
                  onClick={(event) => event.stopPropagation()}
                >
                  <div className="inference-config-modal-head">
                    <div>
                      <h2 id="inference-config-title">完整 vLLM 配置</h2>
                      <p>服务 {service.id} · 当前为配置快照</p>
                    </div>
                    <button
                      type="button"
                      className="close-btn"
                      aria-label="关闭配置详情"
                      title="关闭配置详情"
                      onClick={() => setShowConfigDetails(false)}
                    >
                      ×
                    </button>
                  </div>
                  <div className="inference-config-modal-grid">
                    <div>
                      <span>Engine</span>
                      <strong>vLLM</strong>
                      <small>固定引擎</small>
                    </div>
                    <div>
                      <span>Base Model</span>
                      <strong>{service.model}</strong>
                      <small>来自训练任务</small>
                    </div>
                    <div>
                      <span>{service.config?.adapterVariant === "final" ? "Final" : "Best"} Adapter</span>
                      <strong>{service.adapter}</strong>
                      <small>创建服务时选择</small>
                    </div>
                    <div>
                      <span>Max model length</span>
                      <strong>{maxModelLength} tokens</strong>
                      <small>启动前配置</small>
                    </div>
                    <div>
                      <span>GPU memory utilization</span>
                      <strong>{gpuMemoryUtilization}</strong>
                      <small>显存预算比例</small>
                    </div>
                    <div>
                      <span>KV Cache 分配</span>
                      <strong>
                        {config?.kvCacheMode === "manual"
                          ? `${config.kvCacheMemoryGiB ?? "—"} GiB`
                          : "自动计算"}
                      </strong>
                      <small>启动后以 vLLM 实际值为准</small>
                    </div>
                    <div>
                      <span>KV Cache dtype</span>
                      <strong>{config?.kvCacheDtype ?? "auto"}</strong>
                      <small>跟随模型与硬件</small>
                    </div>
                    <div>
                      <span>Max sequences</span>
                      <strong>{maxNumSeqs}</strong>
                      <small>动态批处理边界</small>
                    </div>
                    <div>
                      <span>Max batched tokens</span>
                      <strong>{maxNumBatchedTokens}</strong>
                      <small>单轮 Token 上限</small>
                    </div>
                    <div>
                      <span>Prefix Caching</span>
                      <strong>{prefixCaching ? "开启" : "关闭"}</strong>
                      <small>可复用相同前缀</small>
                    </div>
                    <div>
                      <span>Chunked Prefill</span>
                      <strong>{chunkedPrefill ? "开启" : "关闭"}</strong>
                      <small>长 Prompt 分块预填充</small>
                    </div>
                    <div>
                      <span>Tensor Parallel</span>
                      <strong>{config?.tensorParallelSize ?? 1}</strong>
                      <small>MVP 单卡</small>
                    </div>
                  </div>
                  <div className="inference-config-modal-note">
                    PagedAttention、连续批处理属于 vLLM
                    底层能力，当前只读展示；KV Cache
                    实际容量和加速指标请查看“性能与缓存”。
                  </div>
                </div>
              </div>
            )}
          </section>
          <section className="panel inference-performance-card">
            <div className="panel-head">
              <div>
                <h2>性能与缓存</h2>
                <p>底层能力只读展示，实际数据来自服务运行时</p>
              </div>
              <span className="inference-runtime-badge">
                {service.state === "running" ? "实时指标" : "等待启动"}
              </span>
            </div>
            <div className="inference-capability-strip">
              <div>
                <span>PagedAttention</span>
                <strong>已启用</strong>
              </div>
              <div>
                <span>连续批处理</span>
                <strong>已启用</strong>
              </div>
              <div>
                <span>Chunked Prefill</span>
                <strong>已启用</strong>
              </div>
              <div>
                <span>Prefix Cache</span>
                <strong>已启用</strong>
              </div>
            </div>
            <div className="inference-runtime-metrics">
              <div>
                <span>KV Cache 容量</span>
                <strong>
                  {service.state === "running" ? "8.4 GiB" : "--"}
                </strong>
                <small>启动后由 vLLM 校准</small>
              </div>
              <div>
                <span>KV Cache Token 容量</span>
                <strong>{service.state === "running" ? "32,768" : "--"}</strong>
                <small>可容纳上下文 Token</small>
              </div>
              <div>
                <span>当前使用率</span>
                <strong>{service.state === "running" ? "62%" : "--"}</strong>
                <small>已用 / 总 Cache Block</small>
              </div>
              <div>
                <span>Prefix Cache 命中率</span>
                <strong>{service.state === "running" ? "38%" : "--"}</strong>
                <small>相同前缀复用比例</small>
              </div>
              <div>
                <span>运行中请求</span>
                <strong>{service.state === "running" ? "2" : "0"}</strong>
                <small>当前生成中的请求</small>
              </div>
              <div>
                <span>等待队列</span>
                <strong>{service.state === "running" ? "5" : "0"}</strong>
                <small>等待进入动态批次</small>
              </div>
              <div>
                <span>平均首 Token</span>
                <strong>{service.state === "running" ? "380 ms" : "--"}</strong>
                <small>TTFT</small>
              </div>
              <div>
                <span>Decode 吞吐</span>
                <strong>
                  {service.state === "running" ? "31.4 tok/s" : "--"}
                </strong>
                <small>生成阶段吞吐</small>
              </div>
            </div>
            <div className="inference-metric-note">
              创建前只展示预计配置，服务启动后以 vLLM 实际 KV Cache、Block
              使用率和请求指标为准。
            </div>
          </section>
        </div>
      </>
    );
  }

  function renderDebug(service: InferenceService) {
    return (
      <div className="inference-debug-layout">
        <section className="panel inference-chat-panel">
          <div className="inference-chat-head">
            <div>
              <h2>在线调试</h2>
              <p>
                多轮对话 · Base Model + {service.config?.adapterVariant === "final" ? "Final" : "Best"} Adapter · {chatParameters.stream ? "SSE 流式输出" : "JSON 完整响应"}
              </p>
            </div>
            <span
              className={
                "inference-live-dot " +
                (service.state === "running" ? "live" : "")
              }
            >
              {service.state === "running" ? "服务可用" : "服务未运行"}
            </span>
          </div>
          <div className="inference-chat-thread" ref={chatThreadRef}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={"inference-chat-turn " + message.role}
              >
                <span className="inference-chat-avatar">
                  {message.role === "user" ? "你" : "AI"}
                </span>
                <div>
                  <p>{message.content || "▍"}</p>
                  {message.meta && <small>{message.meta}</small>}
                </div>
              </div>
            ))}
          </div>
          <div className="inference-chat-composer">
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  sendMessage();
                }
              }}
              placeholder={
                service.state === "running"
                  ? "输入测试问题，Enter 发送，Shift + Enter 换行"
                  : "服务未运行，仍可输入问题；启动服务后发送"
              }
            />
            <button
              className="primary-btn"
              onClick={sendMessage}
              disabled={
                service.state !== "running" || sending || !prompt.trim()
              }
            >
              {sending ? "生成中" : "发送"}
            </button>
          </div>
          {service.state !== "running" && (
            <div className="inference-chat-status-note">
              当前服务未运行，可以先准备测试问题；启动服务后即可发送。
            </div>
          )}
        </section>
        <aside className="panel inference-debug-settings">
          <div className="panel-head">
            <div>
              <h2>本次生成参数</h2>
              <p>仅影响本次调试请求，不修改服务部署配置</p>
            </div>
          </div>
          <div className="inference-field">
            <label>Maximum tokens</label>
            <input
              type="number"
              min="1"
              max={chatOutputLimit}
              value={chatParameters.maxTokens}
              onChange={(event) =>
                setChatParameters((current) => ({
                  ...current,
                  maxTokens: Math.min(
                    Math.max(1, Number(event.target.value) || 1),
                    chatOutputLimit,
                  ),
                }))
              }
            />
            <small>最大不超过服务 Base Model 上下文上限：{chatOutputLimit} tokens</small>
          </div>
          <div className="inference-field">
            <label>Temperature</label>
            <input
              type="number"
              min="0"
              max="2"
              step="0.1"
              value={chatParameters.temperature}
              onChange={(event) =>
                setChatParameters((current) => ({
                  ...current,
                  temperature: Math.min(
                    2,
                    Math.max(0, Number(event.target.value) || 0),
                  ),
                }))
              }
            />
          </div>
          <div className="inference-field">
            <label>Top P</label>
            <input
              type="number"
              min="0.001"
              max="1"
              step="0.05"
              value={chatParameters.topP}
              onChange={(event) =>
                setChatParameters((current) => ({
                  ...current,
                  topP: Math.min(
                    1,
                    Math.max(0.001, Number(event.target.value) || 0.001),
                  ),
                }))
              }
            />
          </div>
          <label className="toggle-row setting-toggle">
            <span>
              <strong>流式输出</strong>
              <small>通过 SSE 展示生成过程</small>
            </span>
            <input
              type="checkbox"
              checked={chatParameters.stream}
              onChange={(event) =>
                setChatParameters((current) => ({
                  ...current,
                  stream: event.target.checked,
                }))
              }
            />
            <i />
          </label>
          <div className="inference-debug-tip">
            <strong>调试说明</strong>
            <p>
              上线与否不影响当前页面调试；停止服务后将无法发送，但可以先编辑问题。
            </p>
          </div>
          <button
            className="secondary-btn inference-clear-chat"
            onClick={() => setMessages([])}
          >
            清空对话
          </button>
        </aside>
      </div>
    );
  }

  function renderApi(service: InferenceService) {
    const endpoint = serviceApiEndpoint(service);
    const curlCommand = serviceCurlCommand(service);
    return (
      <div className="inference-api-layout">
        <section className="panel inference-api-main">
          <div className="panel-head">
            <div>
              <h2>服务 API 调用</h2>
              <p>通过平台代理访问该服务，兼容 OpenAI Chat Completions。</p>
            </div>
            {exposureBadge(service)}
          </div>
          <div className="inference-endpoint">
            <span>POST</span>
            <code>{endpoint}</code>
            <button onClick={() => copyEndpoint(service)}>{copied ? "已复制" : "复制"}</button>
          </div>
          <div className="inference-code-head">
            <strong>cURL · SSE 流式调用</strong>
            <span>需使用当前登录会话或内部网关鉴权</span>
          </div>
          <pre className="inference-code">
            <code>{curlCommand}</code>
          </pre>
          <div className="inference-api-note">
            <strong>使用说明</strong>
            <p>
              请求地址是平台代理接口，不是优云实例的公网地址。当前接口仅对已登录会话或内部网关开放，
              暂未提供公网 API Key。命令行调用时请在同一内网环境执行，并携带登录后的会话 Cookie；
              设置 <code>"stream": false</code> 可返回单个 JSON 响应。
            </p>
          </div>
          <div className="inference-api-note">
            <strong>请求体</strong>
            <p>
              <code>messages</code> 必填；<code>maxTokens</code>、<code>temperature</code>、
              <code>topP</code> 和 <code>stream</code> 使用平台字段名。服务模型由服务配置决定，
              不需要在请求体中传 <code>model</code>。
            </p>
          </div>
        </section>
        <aside className="panel inference-api-side">
          <h2>接口信息</h2>
          <div className="inference-summary-list">
            <div>
              <span>协议</span>
              <strong>HTTP POST / SSE</strong>
            </div>
            <div>
              <span>兼容格式</span>
              <strong>OpenAI API</strong>
            </div>
            <div>
              <span>服务 ID</span>
              <strong>{service.id}</strong>
            </div>
            <div>
              <span>当前状态</span>
              <strong>{stateMeta[service.state].label}</strong>
            </div>
            <div>
              <span>可见范围</span>
              <strong>登录会话 / 内部网络</strong>
            </div>
          </div>
          <div className="inference-api-health">
            <span className={service.state === "running" ? "healthy" : ""} />
            <div>
              <strong>接口可用性</strong>
              <p>
                {service.state === "running"
                  ? "服务运行中，可发送请求"
                  : "服务未运行，接口不可用"}
              </p>
            </div>
          </div>
        </aside>
      </div>
    );
  }

  function renderLogs(service: InferenceService) {
    const logs = logsByService[service.id] ?? [];
    const loading = logsLoading[service.id];
    const loadingOlder = olderLogsLoading[service.id];
    const hasOlder = hasOlderLogs[service.id];
    const error = logsError[service.id];
    return (
      <section className="panel inference-logs-panel">
        <div className="panel-head">
          <div>
            <h2>运行日志</h2>
            <p>
              {service.state === "failed" || service.state === "stopped" || service.state === "archived"
                ? "展示启动、恢复、请求与资源释放过程 · 日志已保存"
                : logConnection === "live"
                  ? "展示启动、恢复、请求与资源释放过程 · WebSocket 实时同步"
                  : logConnection === "fallback"
                    ? "展示启动、恢复、请求与资源释放过程 · WebSocket 断开，已降级为每 5 秒轮询"
                    : "展示启动、恢复、请求与资源释放过程 · 正在连接实时日志"}
            </p>
          </div>
          <div className="form-actions">
            <button
              className="secondary-btn compact-btn"
              onClick={() => downloadServiceLogs(service.id)}
            >
              下载日志
            </button>
            <button
              className="secondary-btn compact-btn"
              disabled={loading}
              onClick={() => void fetchServiceLogs(service.id)}
            >
              {loading ? "读取中..." : "刷新"}
            </button>
          </div>
        </div>
        {error && <div className="form-error">{error}</div>}
        <pre
          className="inference-log-viewer"
          ref={logViewerRef}
          onScroll={(event) => handleLogScroll(service.id, event)}
        >
          {logs.length
            ? logs.join("\n")
            : loading
              ? "正在读取远程运行日志..."
              : "暂无运行日志；服务启动后将实时显示后端与远程 vLLM 日志。"}
        </pre>
        <div className="inference-log-footer">
          <span>
            {loadingOlder
              ? "正在加载更早日志..."
              : hasOlder
                ? `已加载 ${logs.length} 行 · 还有更早日志`
                : `已加载完整日志 · ${logs.length} 行`}
          </span>
          <span>服务：{service.id}</span>
        </div>
      </section>
    );
  }

  function renderDetail() {
    const service = selectedService;
    if (!service) return renderList();
    const isArchived = service.state === "archived";
    const canArchive = ![
      "starting",
      "running",
      "stopping",
      "recovering",
      "archived",
    ].includes(service.state);
    const stageLabels: Record<string, string> = {
      starting: "正在启动",
      preparing: "准备资源",
      creating_instance: "创建优云实例",
      waiting_instance: "等待实例 Running",
      instance_running: "实例已 Running",
      waiting_ssh: "等待 SSH 可用",
      ssh_connecting: "连接 SSH",
      ssh_connected: "SSH 已连接",
      diagnosing_vllm_runtime: "诊断远端运行环境",
      checking_vllm_runtime: "检查远端 vLLM 环境",
      installing_vllm: "安装远端 vLLM",
      vllm_runtime_ready: "远端 vLLM 环境就绪",
      checking_runtime: "检查远端运行环境",
      uploading_adapter: "上传 Adapter",
      adapter_uploaded: "Adapter 已上传",
      starting_vllm: "启动 vLLM",
      vllm_starting: "启动 vLLM",
      loading_model: "加载模型",
      health_check: "健康检查",
      health_check_passed: "健康检查通过",
      health_check_failed: "健康检查失败",
      running: "运行中",
      failed: "启动失败",
    };
    const startup = service.config?.startup;
    const stageKey = service.startupStage ?? startup?.stage ?? (service.state === "starting" || service.state === "recovering" ? "preparing" : service.state);
    const stageLabel = stageLabels[stageKey] ?? stageKey;
    const progress = service.startupProgress == null
      ? (service.state === "running" ? 100 : service.state === "failed" ? 100 : service.state === "starting" || service.state === "recovering" ? 8 : 0)
      : Math.max(0, Math.min(100, service.startupProgress));
    const elapsedSeconds = service.startupElapsedSeconds ?? startup?.elapsedSeconds ?? null;
    const startedAt = service.startupStartedAt ?? startup?.startedAt ?? null;
    const elapsed = elapsedSeconds != null
      ? elapsedSeconds
      : startedAt
        ? Math.max(0, Math.floor((Date.now() - Date.parse(startedAt)) / 1000))
        : null;
    return (
      <section className={`content inference-page${tab === "logs" ? " inference-log-page" : ""}`}>
        <BackButton
          label="返回推理服务"
          onClick={() => {
            setNotice("");
            setView("list");
          }}
        />
        <div className="inference-detail-head">
          <div>
            <div className="inference-detail-title">
              <h1>{service.name}</h1>
              {statusBadge(service)}
              {exposureBadge(service)}
            </div>
            <p>
              {service.description} · {service.id}
            </p>
          </div>
          <div className="form-actions">
            {isArchived ? (
              <>
                <button
                  className="secondary-btn"
                  onClick={() => void restoreService(service.id)}
                >
                  恢复服务
                </button>
                <button
                  className="danger-btn"
                  onClick={() => requestDelete(service.id)}
                >
                  删除服务
                </button>
              </>
            ) : (
              <>
                {service.state === "running" && (
                  <button disabled={!localRuntimeEnabled && serviceProvider(service) === "local"} title={!localRuntimeEnabled && serviceProvider(service) === "local" ? LOCAL_RUNTIME_DISABLED : undefined}
                    className="secondary-btn"
                    onClick={() => void restartService(service.id)}
                  >
                    重启
                  </button>
                )}
                {service.state === "running" ? (
                  <button
                    className="danger-btn"
                    onClick={() => void stopService(service.id)}
                  >
                    停止服务
                  </button>
                ) : service.state === "starting" ? (
                  <button className="danger-btn" onClick={() => void stopService(service.id)}>取消启动</button>
                ) : service.state === "recovering" ? (
                  <>
                    <button className="secondary-btn" disabled>重试中</button>
                    <button className="danger-btn" onClick={() => void stopService(service.id)}>取消重试</button>
                  </>
                ) : service.state === "failed" ? (
                  <button disabled={!localRuntimeEnabled && serviceProvider(service) === "local"} title={!localRuntimeEnabled && serviceProvider(service) === "local" ? LOCAL_RUNTIME_DISABLED : undefined}
                    className="secondary-btn"
                    onClick={() => void recoverService(service.id)}
                  >
                    重试
                  </button>
                ) : service.state === "stopping" ? null : (
                  <button disabled={!localRuntimeEnabled && serviceProvider(service) === "local"} title={!localRuntimeEnabled && serviceProvider(service) === "local" ? LOCAL_RUNTIME_DISABLED : undefined}
                    className="secondary-btn"
                    onClick={() => void startService(service.id)}
                  >
                    启动服务
                  </button>
                )}
                {canArchive && (
                  <button
                    className="secondary-btn"
                    onClick={() => requestArchive(service.id)}
                  >
                    归档
                  </button>
                )}
                {service.state === "running" && (
                  <button
                    className="primary-btn"
                    onClick={() => void toggleExposure(service)}
                  >
                    {service.exposure === "online" ? "下线" : "上线"}
                  </button>
                )}
              </>
            )}
          </div>
        </div>
        {notice && (
          <div className="inference-inline-notice">
            {notice}
            <button onClick={() => setNotice("")}>×</button>
          </div>
        )}
        {isArchived && (
          <div className="inference-archive-notice">
            <strong>该服务已归档</strong>
            <p>
              配置、日志、运行指标和训练产物关联仍会保留；服务不会占用
              GPU，也不能启动、上线或调试。
            </p>
          </div>
        )}
        {(service.state === "starting" || service.state === "recovering") && (
          <div className="inference-startup-progress">
            <div className="inference-startup-progress-inline">
              <div className="inference-startup-progress-label">
                <strong>{stageLabel}</strong>
                <span>{elapsed == null ? "已等待时间未知" : `已等待 ${Math.floor(Math.round(elapsed) / 60)} 分 ${Math.round(elapsed) % 60} 秒`}</span>
              </div>
              <div className="inference-startup-progress-track" role="progressbar" aria-label="启动进度" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}><span style={{ width: `${progress}%` }} /></div>
              <b>{progress}%</b>
            </div>
          </div>
        )}
        {serviceProvider(service) === "local" &&
          activeLocalService &&
          activeLocalService.id !== service.id &&
          !activeStates.includes(service.state) &&
          !isArchived && (
            <div className="inference-resource-conflict">
              <span>!</span>
              <div>
                <strong>本机 GPU 正由“{activeLocalService.name}”占用</strong>
                <p>
                  当前本地服务的启动或重试会被拒绝，请先停止正在运行的本地服务。
                </p>
              </div>
            </div>
          )}
        <div className="inference-tabs">
          <button
            className={tab === "overview" ? "active" : ""}
            onClick={() => setTab("overview")}
          >
            服务概览
          </button>
          {!isArchived && (
            <>
              <button
                className={tab === "debug" ? "active" : ""}
                onClick={() => setTab("debug")}
              >
                在线调试
              </button>
              <button
                className={tab === "api" ? "active" : ""}
                onClick={() => setTab("api")}
              >
                API 使用
              </button>
            </>
          )}
          <button
            className={tab === "logs" ? "active" : ""}
            onClick={() => setTab("logs")}
          >
            运行日志
          </button>
        </div>
        {tab === "overview" && renderOverview(service)}
        {tab === "debug" && !isArchived && renderDebug(service)}
        {tab === "api" && !isArchived && renderApi(service)}
        {tab === "logs" && renderLogs(service)}
      </section>
    );
  }

  const content =
    view === "list"
      ? renderList()
      : view === "create"
        ? renderCreate()
        : renderDetail();
  return (
    <>
      {content}
      {retryPicker && (
        <div
          className="inference-config-modal-backdrop"
          onClick={() => setRetryPicker(null)}
        >
          <div
            ref={retryDialogRef}
            className="inference-config-modal inference-retry-modal"
            tabIndex={-1}
            role="dialog"
            aria-modal="true"
            aria-labelledby="inference-retry-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="inference-config-modal-head">
              <div>
                <h2 id="inference-retry-title">重新选择优云 GPU</h2>
                <p>原配置无库存，仅调整 GPU，其他配置保持不变。</p>
              </div>
              <button type="button" className="close-btn" aria-label="关闭 GPU 选择" title="关闭 GPU 选择" onClick={() => setRetryPicker(null)}><X size={18} aria-hidden="true" /></button>
            </div>
            {retryPicker.error && <div className="inference-resource-warning"><strong>库存查询失败</strong><p>{retryPicker.error}</p></div>}
            <div className="retry-picker-section">
              <span>GPU 型号</span>
              <div className="youyun-machine-list">
                {(retryPicker.specs?.machineTypes ?? []).map((machine) => (
                  <button
                    key={machine.name}
                    type="button"
                    className={`youyun-machine-item${youyunMachineIsUnavailable(machine, "inference") ? " sold-out" : ""}${retryPicker.gpuType === machine.name ? " selected" : ""}`}
                    disabled={retryPicker.loading || youyunMachineIsUnavailable(machine, "inference")}
                    onClick={() => void loadRetrySpecs(machine.name)}
                  >
                    <strong>{machine.name}</strong>
                    <small>{youyunMachineInventoryLabel(machine)}</small>
                  </button>
                ))}
              </div>
            </div>
            <div className="retry-picker-section">
              <span>单卡规格</span>
              <div className="youyun-capacity-list">
                {(retryPicker.specs?.capacity ?? []).filter((item) => item.gpu === 1).map((item) => (
                  <button
                    key={`${item.cpu}-${item.memoryGb}`}
                    type="button"
                    className={`youyun-capacity-item${retryPicker.cpu === String(item.cpu) && retryPicker.memory === String(item.memoryGb) ? " selected" : ""}`}
                    disabled={retryPicker.loading || !item.available}
                    onClick={() => setRetryPicker((value) => value ? { ...value, cpu: String(item.cpu), memory: String(item.memoryGb) } : value)}
                  >
                    {item.cpu} 核 · {item.memoryGb}GB{!item.available ? " · 无库存" : ""}
                  </button>
                ))}
              </div>
            </div>
            <div className="inference-config-modal-note">将继续使用原模型、Adapter、上下文长度、并发和推理参数。</div>
            <div className="modal-actions">
              <button type="button" className="secondary-btn" onClick={() => setRetryPicker(null)}>取消</button>
              <button className="primary-btn" disabled={retryPicker.loading || !retryPicker.cpu || !retryPicker.memory} onClick={() => void submitRetryWithResource()}>
                {retryPicker.loading ? "查询中..." : "确认重试"}
              </button>
            </div>
          </div>
        </div>
      )}
      {confirmAction && (
        <div
          className="inference-confirm-backdrop"
          onClick={() => setConfirmAction(null)}
        >
          <div
            ref={confirmDialogRef}
            className="inference-confirm-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="inference-confirm-title"
            tabIndex={-1}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="inference-confirm-icon">!</div>
            <div>
                <h2 id="inference-confirm-title">
                {confirmAction.type === "archive"
                  ? "确认归档服务"
                  : "确认永久删除服务"}
              </h2>
              <p>
                {confirmAction.type === "archive"
                  ? "归档后该服务会从默认列表隐藏，配置、日志、运行指标和训练产物关联仍会保留。服务不会继续占用 GPU。"
                  : "删除后服务配置、日志和运行指标将永久删除，但不会删除训练任务和 Adapter。此操作不可恢复。"}
              </p>
            </div>
            <div className="inference-confirm-actions">
              <button
                className="secondary-btn"
                onClick={() => setConfirmAction(null)}
              >
                取消
              </button>
              <button
                className={
                  confirmAction.type === "delete" ? "danger-btn" : "primary-btn"
                }
                onClick={confirmServiceAction}
              >
                {confirmAction.type === "archive" ? "确认归档" : "永久删除"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
