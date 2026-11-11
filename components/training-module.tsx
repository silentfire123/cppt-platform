"use client";

import { useCapabilities, LOCAL_RUNTIME_DISABLED } from "../lib/use-capabilities";

import { KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, FileText, RefreshCw, X } from "lucide-react";
import { ChatMessage, ChatParameters, ChatResponse, Dataset, DatasetPrompt, TrainingMetric, TrainingParameters, TrainingState, TrainingTask, TrainingTaskType, TuningMode } from "../lib/types";
import { apiUrl, apiWebSocketUrl } from "../lib/api-config";
import { formatApiError } from "../lib/api-error";
import { normalizeInferenceLogs } from "../lib/inference-logs";
import { clearSessionDraft, readSessionDraft, TRAINING_DRAFT_STORAGE_KEY, writeSessionDraft } from "../lib/interface-state";
import { ApiRequestError, backendStateAfterSubmitError } from "../lib/training-submit-state";
import { trainingArtifactInfo } from "../lib/training-artifact-presentation";
import { mergeTrainingDetail } from "../lib/training-detail-state";
import { cachedYouyunRequest, friendlyYouyunError, loadYouyunSpecs, YOUYUN_IMAGES_CACHE_TTL_MS, youyunImageRuntimeLabel, youyunInventoryMessage, youyunMachineInventoryClass, youyunMachineInventoryLabel, youyunMachineIsUnavailable, youyunPreferredMode, youyunRegionLabel, type YouyunLocationOption, type YouyunLocationMode } from "../lib/youyun-location";
import { BackButton } from "./back-button";
import { Pagination } from "./pagination";
import type { TrainingCreateRequest } from "../lib/model-handoff";
import { trainingModelCatalog } from "../lib/training-model-source";

type TrainingView = "list" | "create" | "detail" | "baseline-chat" | "compare";
type ExecutionMode = "web" | "cli";
type ExecutionEnvironment = "local" | "youyun";
type BackendExecutionMode = "local" | "remote";
type YouyunInstanceMode = "uhost" | "pod";
type DistributedStrategy = "single" | "ddp" | "deepspeed_zero2";
type YouyunGpuSpecs = {
  region: string;
  zone: string;
  instanceMode: YouyunInstanceMode;
  gpuType: string;
  gpuMemoryGb?: number | null;
  defaultImageId?: string | null;
  chargeType: string;
  machineTypes: { name: string; zone: string; soldOut: boolean; description: string; graphicsMemoryGb: number | null; inventoryStatus?: "available" | "sold_out" | "unknown" | "error"; inventoryMessage?: string; inventoryError?: string; trainingSupported?: boolean; supportReason?: string | null }[];
  capacity: { gpu: number; cpu: number; memoryGb: number; available: boolean }[];
  zones: { region: string; zone: string; describe: string; is_pod: boolean }[];
  locations?: YouyunLocationOption[];
  locationZone?: string;
  imageType?: string | null;
  container?: boolean | null;
  selectedGpuCount?: number;
  selectedDistributedStrategy?: DistributedStrategy;
};
type YouyunImage = { id: string; name: string; type: string; status: string; tags: string[]; description: string; source?: string; instanceMode?: "pod" | "uhost"; container?: boolean | null; isDefault?: boolean };
type YouyunResource = {
  mode: "auto" | "manual";
  gpuType: string;
  gpuCount: number;
  distributedStrategy?: DistributedStrategy;
  cpuCores: number;
  memoryGb: number;
  region: string;
  zone: string;
  instanceMode: YouyunInstanceMode;
  chargeType: string;
  price: { currency: string; instance: number; disks: number; systemDisks: number; image: number; total: number };
};
type BackendState = "checking" | "online" | "offline";
type LogConnectionState = "connecting" | "live" | "fallback";
type ModelOption = { id: string; label: string; size: string; cached: boolean; recommended?: boolean; publicModelPath?: string; source?: "youyun-public" | "modelscope"; tag?: string | null };
type ModelSearchResult = { id: string; label: string; fileSize: number; downloads: number; cached: boolean; publicModelPath?: string; source?: "youyun-public" | "modelscope"; tag?: string | null; size?: string | null };
type ArtifactPreviewFile = { path: string; sizeBytes: number; modifiedAt?: string };
type ArtifactPreview = { path: string; name: string; files: ArtifactPreviewFile[]; truncated?: boolean };
type ArtifactPreviewFileResult = { path: string; sizeBytes: number; modifiedAt?: string; kind?: "text" | "binary"; content?: string; truncated?: boolean; message?: string };

const taskStateMeta: Record<TrainingState, { label: string; className: string }> = {
  queued: { label: "排队中", className: "status-queued" },
  provisioning: { label: "准备资源", className: "status-running" },
  preparing: { label: "准备训练", className: "status-running" },
  running: { label: "运行中", className: "status-running" },
  stopping: { label: "停止中", className: "status-stopped" },
  collecting: { label: "收集产物", className: "status-running" },
  uploading_artifact: { label: "上传产物", className: "status-running" },
  completed: { label: "已完成", className: "status-completed" },
  failed: { label: "失败", className: "status-failed" },
  stopped: { label: "已停止", className: "status-stopped" }
};

function formatModelBytes(bytes: number) {
  if (!bytes) return "提交训练时从 ModelScope 下载";
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function compactLocalDate(date = new Date()) {
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("");
}

function modelShortName(model: string) {
  return model.includes("/") ? model.split("/").pop()! : model;
}

function youyunImageLabel(imageId: string, images: YouyunImage[]) {
  return images.find((item) => item.id === imageId)?.name
    || (imageId === "compshareImage-1bvy5l0kjip2" ? "LLaMA-Factory 社区训练镜像" : imageId);
}

function tuningModeLabel(mode: TuningMode) {
  return mode === "qlora" ? "QLoRA" : "LoRA";
}

function suggestTaskName(model: string, dataset: Dataset | undefined, taskType: TrainingTaskType, mode: TuningMode) {
  const modelName = modelShortName(model) || "Qwen2.5";
  const datasetName = dataset?.name || "语料";
  return `${modelName} ${datasetName} ${taskType} ${tuningModeLabel(mode)} 实验`;
}

function suggestOutputName(model: string, taskType: TrainingTaskType, mode: TuningMode) {
  const suffix = `${taskType.toLowerCase()}-${mode}`;
  const repoName = modelShortName(model);
  return `${(repoName || "my-model").toLowerCase()}-${suffix}-${compactLocalDate()}`;
}

async function apiRequest<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = options?.body
    ? { "Content-Type": "application/json", ...(options.headers ?? {}) }
    : options?.headers;
  let response: Response;
  try {
    response = await fetch(apiUrl(path), {
      ...options,
      credentials: "include",
      headers,
      cache: "no-store"
    });
  } catch {
    throw new ApiRequestError("训练服务连接失败，请检查后端服务。", null);
  }
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const body = await response.json();
      message = formatApiError(body, message);
    } catch {
      // Keep the HTTP fallback when the backend did not return JSON.
    }
    throw new ApiRequestError(message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

function SuggestedNameInput({
  value,
  suggestion,
  onChange,
}: {
  value: string;
  suggestion: string;
  onChange: (value: string) => void;
}) {
  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key !== "Tab" || value.trim() || !suggestion) return;
    event.preventDefault();
    onChange(suggestion);
  }

  return (
    <div className="suggested-name-input">
      <input
        className="input"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={`例如：${suggestion}`}
      />
      {!value.trim() && suggestion && (
        <span className="suggested-name-hint"><kbd>Tab</kbd> 应用建议</span>
      )}
    </div>
  );
}

const defaultParameters: TrainingParameters = {
  epochs: 3,
  learningRate: "2e-4",
  batchSize: 1,
  gradientAccumulation: 8,
  validationSteps: 50,
  maxSequenceLength: 1024,
  loraRank: 16,
  loraAlpha: 16,
  loraDropout: 0,
  lrScheduler: "linear",
  loraPlusEnabled: false,
  loraPlusRatio: 16,
  targetModules: "auto",
  optimizer: "adamw_torch",
  warmupRatio: 0.1,
  weightDecay: 0.01,
  maxGradNorm: 1,
  seed: 42,
  loggingSteps: 10,
  packing: false,
  gradientCheckpointing: true
};

const defaultChatParameters: ChatParameters = {
  maxNewTokens: 256,
  temperature: 0.2,
  topP: 0.9,
  seed: 42
};

function taskActions(state: TrainingState) {
  if (["running", "provisioning", "preparing", "collecting", "uploading_artifact"].includes(state)) return ["详细", "停止"];
  if (state === "stopping") return ["详细"];
  if (state === "queued") return ["详细", "停止", "删除"];
  if (state === "failed" || state === "stopped") return ["详细", "重试", "克隆", "删除"];
  return ["详细", "克隆", "删除"];
}

function deploymentHint(task: TrainingTask) {
  if (task.state !== "completed" && task.state !== "stopped") return task.currentStep;
  const hasBestAdapter = Boolean(task.bestAdapterPath) || (task.artifacts?.some((artifact) => artifact.type === "adapter" && artifact.variant === "best" && artifact.path) ?? false);
  const hasAdapter = task.artifacts?.some((artifact) => artifact.type === "adapter" && artifact.path) ?? Boolean(task.selectedAdapterPath || task.finalAdapterPath);
  const prefix = task.state === "stopped" ? "训练已停止" : "训练完成";
  if (hasBestAdapter) return `${prefix}，Best Adapter 已保存，可创建推理服务`;
  if (hasAdapter) return `${prefix}，Final Adapter 已保存，可创建推理服务`;
  return `${prefix}，但没有可部署 Adapter`;
}

function trainingNodeCopy(task: TrainingTask) {
  const remote = task.provider === "youyun" || task.executionMode === "remote";
  return remote
    ? { node: "远程优云 GPU 训练节点", worker: "优云 SSH Worker" }
    : { node: "本地 CPU 训练节点", worker: "本地 LLaMA-Factory Worker" };
}

function isActiveTrainingState(state: TrainingState) {
  return ["running", "provisioning", "preparing", "collecting", "uploading_artifact"].includes(state);
}

function isTaskPollingState(state: TrainingState) {
  return ["queued", "provisioning", "preparing", "running", "stopping", "collecting", "uploading_artifact"].includes(state);
}

export function TrainingModule({
  datasets,
  datasetsLoaded,
  datasetsLoading,
  datasetsError,
  onLoadDatasets,
  onOpenDatasets,
  createRequest,
  onCreateRequestHandled,
}: {
  datasets: Dataset[];
  datasetsLoaded: boolean;
  datasetsLoading: boolean;
  datasetsError: string;
  onLoadDatasets: () => Promise<void>;
  onOpenDatasets: () => void;
  createRequest?: TrainingCreateRequest | null;
  onCreateRequestHandled?: (requestId: number) => void;
}) {
  const { localRuntimeEnabled, capabilities } = useCapabilities();
  const handledCreateRequestRef = useRef<number | null>(null);
  const [view, setView] = useState<TrainingView>("list");
  const [draftHydrated, setDraftHydrated] = useState(false);
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [taskPage, setTaskPage] = useState(1);
  const [taskPageInfo, setTaskPageInfo] = useState({ pageSize: 5, total: 0, totalPages: 1 });
  const [selectedId, setSelectedId] = useState("");
  const [detailTask, setDetailTask] = useState<TrainingTask | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [detailLogConnection, setDetailLogConnection] = useState<LogConnectionState>("connecting");
  const [backendState, setBackendState] = useState<BackendState>("checking");
  const [submitting, setSubmitting] = useState(false);
  const [query, setQuery] = useState("");
  const [stateFilter, setStateFilter] = useState("all");
  const [taskName, setTaskName] = useState("");
  const [taskType, setTaskType] = useState<TrainingTaskType>("SFT");
  const [executionMode, setExecutionMode] = useState<ExecutionMode>("web");
  const [environment, setEnvironment] = useState<ExecutionEnvironment>("youyun");
  useEffect(() => {
    if (capabilities && !localRuntimeEnabled) { setEnvironment("youyun"); setExecutionMode("web"); }
  }, [capabilities, localRuntimeEnabled, environment, executionMode]);
  const [youyunSpecs, setYouyunSpecs] = useState<YouyunGpuSpecs | null>(null);
  const [selectedYouyunMode, setSelectedYouyunMode] = useState<YouyunInstanceMode>("pod");
  const [selectedYouyunRegion, setSelectedYouyunRegion] = useState("");
  const [selectedYouyunZone, setSelectedYouyunZone] = useState("");
  const [youyunLoading, setYouyunLoading] = useState(false);
  const [youyunError, setYouyunError] = useState("");
  const [tuningMode, setTuningMode] = useState<TuningMode>("qlora");
  const [modelQuery, setModelQuery] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedPublicModelPath, setSelectedPublicModelPath] = useState("");
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([]);
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [modelSearching, setModelSearching] = useState(false);
  const [modelSearchError, setModelSearchError] = useState("");
  const [selectedYouyunGpuType, setSelectedYouyunGpuType] = useState("");
  const [selectedYouyunGpuCount, setSelectedYouyunGpuCount] = useState(1);
  const [distributedStrategy, setDistributedStrategy] = useState<DistributedStrategy>("single");
  const [selectedYouyunCpuCores, setSelectedYouyunCpuCores] = useState<number | null>(null);
  const [selectedYouyunMemoryGb, setSelectedYouyunMemoryGb] = useState<number | null>(null);
  const [youyunResource, setYouyunResource] = useState<YouyunResource | null>(null);
  const [youyunPriceLoading, setYouyunPriceLoading] = useState(false);
  const [youyunPriceError, setYouyunPriceError] = useState("");
  const youyunPriceRequestRef = useRef(0);
  const youyunSpecsRequestRef = useRef(0);
  const [youyunImages, setYouyunImages] = useState<YouyunImage[]>([]);
  const [selectedYouyunImageId, setSelectedYouyunImageId] = useState("");
  const [youyunImagesLoading, setYouyunImagesLoading] = useState(false);
  const [youyunImagesError, setYouyunImagesError] = useState("");
  const [imagePickerOpen, setImagePickerOpen] = useState(false);
  const [pendingYouyunImageId, setPendingYouyunImageId] = useState("");
  const selectedYouyunImage = youyunImages.find((image) => image.id === selectedYouyunImageId);
  const selectedYouyunImageContainer = selectedYouyunImage?.container ?? youyunSpecs?.container;
  const selectedYouyunImageIsContainer = selectedYouyunImageContainer === true;
  const selectedTrainingModelCatalog = trainingModelCatalog({
    environment,
    selectedImageContainer: selectedYouyunImage?.container,
    specsContainer: youyunSpecs?.container,
  });
  const usesYouyunPublicModels = selectedTrainingModelCatalog === "public";
  const publishedTextDatasets = useMemo(() => datasets.filter((dataset) => dataset.state === "published" && dataset.type === "文本生成" && (dataset.purpose ?? "training") === "training"), [datasets]);
  const publishedDatasets = useMemo(() => publishedTextDatasets.filter((dataset) => dataset.structure === taskType), [publishedTextDatasets, taskType]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const selectableDatasets = executionMode === "cli" ? publishedTextDatasets : publishedDatasets;
  const exactDataset = selectableDatasets.find((dataset) => dataset.id === selectedDatasetId);
  const selectedDataset = executionMode === "cli" ? exactDataset ?? selectableDatasets[0] : exactDataset;
  const taskNameSuggestion = suggestTaskName(selectedModel, selectedDataset, taskType, tuningMode);
  const outputNameSuggestion = suggestOutputName(selectedModel, taskType, tuningMode);
  const [mergeModel, setMergeModel] = useState(false);
  const [validationSplit, setValidationSplit] = useState(10);
  const [earlyStopping, setEarlyStopping] = useState(false);
  const [patience, setPatience] = useState(3);
  const [minDelta, setMinDelta] = useState(0);
  const [outputName, setOutputName] = useState("");
  const [checkpointLimit, setCheckpointLimit] = useState(2);
  const [checkpointStrategy, setCheckpointStrategy] = useState<"epoch" | "steps">("epoch");
  const [checkpointInterval, setCheckpointInterval] = useState(100);
  const [maxSteps, setMaxSteps] = useState(-1);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [formError, setFormError] = useState("");
  const [showCreateLeaveConfirm, setShowCreateLeaveConfirm] = useState(false);
  const [cliNotice, setCliNotice] = useState("");
  const [cliCommandText, setCliCommandText] = useState("");
  const [parameters, setParameters] = useState(defaultParameters);
  const [baselineMessages, setBaselineMessages] = useState<ChatMessage[]>([]);
  const [chatParameters, setChatParameters] = useState(defaultChatParameters);
  const preserveModelOnEnvironmentChangeRef = useRef(false);

  useEffect(() => {
    const navigationType = (window.performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined)?.type;
    if (navigationType === "reload") {
      const saved = readSessionDraft<Partial<{ version: number; view: TrainingView; selectedId: string; form: Record<string, unknown> }>>(TRAINING_DRAFT_STORAGE_KEY);
      const form = saved?.form;
      if (saved?.version === 1 && saved.view) setView(saved.view);
      if (saved?.selectedId) setSelectedId(saved.selectedId);
      if (form) {
        if (typeof form.taskName === "string") setTaskName(form.taskName);
        if (form.taskType === "SFT" || form.taskType === "DPO") setTaskType(form.taskType);
        if (form.executionMode === "web" || form.executionMode === "cli") setExecutionMode(form.executionMode);
        if (form.environment === "local" || form.environment === "youyun") setEnvironment(form.environment);
        if (form.tuningMode === "lora" || form.tuningMode === "qlora") setTuningMode(form.tuningMode);
        if (typeof form.modelQuery === "string") setModelQuery(form.modelQuery);
        if (typeof form.selectedModel === "string") setSelectedModel(form.selectedModel);
        if (typeof form.selectedPublicModelPath === "string") setSelectedPublicModelPath(form.selectedPublicModelPath);
        if (typeof form.selectedDatasetId === "string") setSelectedDatasetId(form.selectedDatasetId);
        if (typeof form.selectedYouyunMode === "string") setSelectedYouyunMode(form.selectedYouyunMode as YouyunInstanceMode);
        if (typeof form.selectedYouyunRegion === "string") setSelectedYouyunRegion(form.selectedYouyunRegion);
        if (typeof form.selectedYouyunZone === "string") setSelectedYouyunZone(form.selectedYouyunZone);
        if (typeof form.selectedYouyunGpuType === "string") setSelectedYouyunGpuType(form.selectedYouyunGpuType);
        // Normalize legacy multi-card drafts to the current single-card boundary.
        setSelectedYouyunGpuCount(1);
        setDistributedStrategy("single");
        if (typeof form.selectedYouyunCpuCores === "number") setSelectedYouyunCpuCores(form.selectedYouyunCpuCores);
        if (typeof form.selectedYouyunMemoryGb === "number") setSelectedYouyunMemoryGb(form.selectedYouyunMemoryGb);
        if (typeof form.selectedYouyunImageId === "string") setSelectedYouyunImageId(form.selectedYouyunImageId);
        if (typeof form.mergeModel === "boolean") setMergeModel(form.mergeModel);
        if (typeof form.validationSplit === "number") setValidationSplit(form.validationSplit);
        if (typeof form.earlyStopping === "boolean") setEarlyStopping(form.earlyStopping);
        if (typeof form.patience === "number") setPatience(form.patience);
        if (typeof form.minDelta === "number") setMinDelta(form.minDelta);
        if (typeof form.outputName === "string") setOutputName(form.outputName);
        if (typeof form.checkpointLimit === "number") setCheckpointLimit(form.checkpointLimit);
        if (form.checkpointStrategy === "epoch" || form.checkpointStrategy === "steps") setCheckpointStrategy(form.checkpointStrategy);
        if (typeof form.checkpointInterval === "number") setCheckpointInterval(form.checkpointInterval);
        if (typeof form.maxSteps === "number") setMaxSteps(form.maxSteps);
        if (form.parameters && typeof form.parameters === "object") setParameters((form.parameters as typeof defaultParameters));
        // File objects are intentionally not persisted. The form will ask the user to reselect them if needed.
      }
    }
    setDraftHydrated(true);
  }, []);

  useEffect(() => {
    if (!draftHydrated) return;
    if (view !== "create") {
      clearSessionDraft(TRAINING_DRAFT_STORAGE_KEY);
      return;
    }
    writeSessionDraft(TRAINING_DRAFT_STORAGE_KEY, {
      version: 1,
      view,
      selectedId,
      form: {
        taskName, taskType, executionMode, environment, tuningMode, modelQuery,
        selectedModel, selectedPublicModelPath, selectedDatasetId, selectedYouyunMode,
        selectedYouyunRegion, selectedYouyunZone, selectedYouyunGpuType,
        selectedYouyunGpuCount: 1, distributedStrategy: "single", selectedYouyunCpuCores, selectedYouyunMemoryGb, selectedYouyunImageId,
        mergeModel, validationSplit, earlyStopping, patience, minDelta, outputName,
        checkpointLimit, checkpointStrategy, checkpointInterval, maxSteps, parameters,
      },
    });
  }, [draftHydrated, view, selectedId, taskName, taskType, executionMode, environment, tuningMode, modelQuery, selectedModel, selectedPublicModelPath, selectedDatasetId, selectedYouyunMode, selectedYouyunRegion, selectedYouyunZone, selectedYouyunGpuType, selectedYouyunGpuCount, distributedStrategy, selectedYouyunCpuCores, selectedYouyunMemoryGb, selectedYouyunImageId, mergeModel, validationSplit, earlyStopping, patience, minDelta, outputName, checkpointLimit, checkpointStrategy, checkpointInterval, maxSteps, parameters]);

  useEffect(() => {
    if (preserveModelOnEnvironmentChangeRef.current) {
      preserveModelOnEnvironmentChangeRef.current = false;
      return;
    }
    setSelectedModel("");
    setSelectedPublicModelPath("");
    setModelQuery("");
    setModelOptions([]);
  }, [environment]);

  useEffect(() => {
    if (
      !createRequest ||
      handledCreateRequestRef.current === createRequest.requestId
    ) {
      return;
    }
    handledCreateRequestRef.current = createRequest.requestId;
    setExecutionMode("web");
    if (createRequest.environment && createRequest.environment !== environment) {
      preserveModelOnEnvironmentChangeRef.current = true;
      setEnvironment(createRequest.environment);
    }
    setSelectedModel(createRequest.model);
    setSelectedPublicModelPath(createRequest.publicModelPath || "");
    setModelQuery(createRequest.model);
    setModelDropdownOpen(false);
    setModelSearchError("");
    setFormError("");
    setView("create");
    onCreateRequestHandled?.(createRequest.requestId);
  }, [createRequest, environment, onCreateRequestHandled]);

  useEffect(() => {
    if (executionMode !== "cli") return;
    if (!selectableDatasets.some((dataset) => dataset.id === selectedDatasetId)) {
      setSelectedDatasetId(selectableDatasets[0]?.id ?? "");
    }
  }, [executionMode, selectableDatasets, selectedDatasetId]);

  useEffect(() => {
    if (executionMode !== "cli" || !selectedDataset || !cliCommandText) return;
    setCliCommandText((command) => updateCliDatasetLines(command, selectedDataset));
  }, [executionMode, selectedDataset?.id, selectedDataset?.name, selectedDataset?.structure]);

  const selectedTask = (detailTask?.id === selectedId ? detailTask : undefined)
    ?? tasks.find((task) => task.id === selectedId)
    ?? tasks[0];
  const filteredTasks = useMemo(() => tasks.filter((task) => {
    const matchesText = task.name.toLowerCase().includes(query.toLowerCase()) || task.id.includes(query);
    return matchesText && (stateFilter === "all" || task.state === stateFilter);
  }), [tasks, query, stateFilter]);
  const shouldPollTasks = tasks.some((task) => isTaskPollingState(task.state));
  const directRepoId = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(modelQuery.trim()) && modelQuery.trim() !== selectedModel && !modelOptions.some((model) => model.id === modelQuery.trim()) ? modelQuery.trim() : "";
  const createNavItems = useMemo(() => {
    const items = [{
      id: "training-section-basic",
      index: "01",
      title: "基本信息",
      summary: `${taskType} · ${tuningModeLabel(tuningMode)} · ${environment === "youyun" ? "远程优云 GPU" : "本地 CPU"}`,
    }];
    if (executionMode === "web" && (selectedModel || selectedDataset)) {
      items.push({
        id: "training-section-model-data",
        index: "02",
        title: "模型与数据",
        summary: [selectedModel || "基础模型待填写", selectedDataset?.name || "训练数据集待填写"].join(" · "),
      });
    }
    if (executionMode === "web" && selectedDataset) {
      items.push(
        { id: "training-section-parameters", index: "03", title: "训练参数", summary: `${parameters.epochs} Epochs · 批次 ${parameters.batchSize} · 梯度累积 ${parameters.gradientAccumulation}` },
        { id: "training-section-validation", index: "04", title: "验证与早停", summary: `${validationSplit}% 验证 · ${earlyStopping ? `早停 Patience ${patience}` : "未启用早停"}` },
        { id: "training-section-output", index: "05", title: "训练产出与检查点", summary: `${outputName.trim() || "模型名称待填写"} · 保留 ${checkpointLimit} 个检查点` },
      );
    }
    return items;
  }, [earlyStopping, environment, executionMode, outputName, parameters.batchSize, parameters.epochs, parameters.gradientAccumulation, patience, selectedDataset, selectedModel, taskType, tuningMode, validationSplit]);

  const createFormDirty = Boolean(
    taskName.trim() ||
    outputName.trim() ||
    selectedModel ||
    selectedDatasetId ||
    cliCommandText.trim() ||
    baselineMessages.length ||
    executionMode !== "web" ||
    taskType !== "SFT" ||
    tuningMode !== "lora" ||
    environment !== "local" ||
    mergeModel ||
    validationSplit !== 10 ||
    earlyStopping ||
    patience !== 3 ||
    minDelta !== 0 ||
    checkpointLimit !== 2 ||
    checkpointStrategy !== "epoch" ||
    checkpointInterval !== 100 ||
    maxSteps !== -1 ||
    selectedYouyunGpuType ||
    selectedYouyunCpuCores !== null ||
    selectedYouyunMemoryGb !== null ||
    selectedYouyunMode !== "pod" ||
    JSON.stringify(parameters) !== JSON.stringify(defaultParameters) ||
    JSON.stringify(chatParameters) !== JSON.stringify(defaultChatParameters),
  );

  function requestCreateBack() {
    if (createFormDirty) {
      setShowCreateLeaveConfirm(true);
      return;
    }
    setView("list");
  }

  function leaveCreate() {
    setShowCreateLeaveConfirm(false);
    clearSessionDraft(TRAINING_DRAFT_STORAGE_KEY);
    setView("list");
  }

  function jumpToCreateSection(id: string) {
    const target = document.getElementById(id);
    const scrollContainer = document.querySelector<HTMLElement>(".main");
    if (!target || !scrollContainer) return;

    // Scroll only the app content pane so browser viewport sizing and scrollbar geometry stay stable.
    const containerRect = scrollContainer.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const nextTop = scrollContainer.scrollTop + targetRect.top - containerRect.top;
    scrollContainer.scrollTo({ top: Math.max(0, nextTop), behavior: "auto" });
  }
  async function refreshYouyunCapacity(gpuType: string, regionOverride?: string, zoneOverride?: string, gpuCountOverride?: number) {
    if (!gpuType) return;
    await refreshYouyunSpecs(gpuType, regionOverride, zoneOverride, undefined, undefined, gpuCountOverride, true);
  }


  const refreshYouyunSpecs = useCallback(async (
    gpuTypeOverride?: string,
    regionOverride?: string,
    zoneOverride?: string,
    imageOverride?: string,
    modeOverride?: YouyunInstanceMode,
    gpuCountOverride?: number,
    force = false,
  ) => {
    const requestId = ++youyunSpecsRequestRef.current;
    ++youyunPriceRequestRef.current;
    setYouyunLoading(true);
    setYouyunError("");
    setYouyunResource(null);
    setYouyunPriceError("");
    setYouyunPriceLoading(false);
    setSelectedYouyunCpuCores(null);
    setSelectedYouyunMemoryGb(null);
    try {
      const specs = await loadYouyunSpecs<YouyunGpuSpecs>(apiRequest, {
        gpuType: gpuTypeOverride ?? selectedYouyunGpuType,
        imageId: imageOverride ?? selectedYouyunImageId,
        region: regionOverride ?? selectedYouyunRegion,
        zone: zoneOverride ?? selectedYouyunZone,
        instanceMode: modeOverride ?? selectedYouyunMode,
      }, "training", () => requestId === youyunSpecsRequestRef.current, { force });
      if (!specs || requestId !== youyunSpecsRequestRef.current) return;
      setYouyunSpecs({ ...specs, selectedGpuCount: gpuCountOverride ?? 1 });
      setSelectedYouyunMode(specs.instanceMode);
      setSelectedYouyunRegion(specs.region);
      setSelectedYouyunZone(specs.locationZone ?? specs.zone);
      setSelectedYouyunGpuType(specs.gpuType);
      if (!selectedYouyunImageId && specs.defaultImageId) setSelectedYouyunImageId(specs.defaultImageId);
      const gpuCount = gpuCountOverride ?? 1;
      const available = specs.capacity.filter((item) => item.gpu === gpuCount && item.available);
      const selected = available.find((item) => item.cpu === selectedYouyunCpuCores && item.memoryGb === selectedYouyunMemoryGb) ?? available[0];
      setSelectedYouyunCpuCores(selected?.cpu ?? null);
      setSelectedYouyunMemoryGb(selected?.memoryGb ?? null);
    } catch (error) {
      if (requestId === youyunSpecsRequestRef.current) setYouyunError(friendlyYouyunError(error instanceof Error ? error.message : "", "capacity"));
    } finally {
      if (requestId === youyunSpecsRequestRef.current) setYouyunLoading(false);
    }
  }, [selectedYouyunGpuType, selectedYouyunCpuCores, selectedYouyunMemoryGb, selectedYouyunImageId, selectedYouyunRegion, selectedYouyunZone, selectedYouyunMode]);

  const refreshYouyunImages = useCallback(async () => {
    setYouyunImagesLoading(true); setYouyunImagesError("");
    try {
      const result = await cachedYouyunRequest<{ images: YouyunImage[]; defaultImageId: string | null }>(apiRequest, "/youyun/images", { ttlMs: YOUYUN_IMAGES_CACHE_TTL_MS });
      setYouyunImages(result.images);
      if (!selectedYouyunImageId || !result.images.some((item) => item.id === selectedYouyunImageId)) {
        const imageId = result.defaultImageId || result.images[0]?.id || "";
        setSelectedYouyunImageId(imageId);
        void refreshYouyunSpecs(undefined, undefined, undefined, imageId);
        setYouyunResource(null);
      }
    } catch (error) {
      setYouyunImagesError(error instanceof Error ? error.message : "优云官方镜像查询失败。");
    } finally { setYouyunImagesLoading(false); }
  }, [selectedYouyunImageId, refreshYouyunSpecs]);

  useEffect(() => {
    if (environment === "youyun" && !youyunSpecs && !youyunLoading && !youyunError) void refreshYouyunSpecs(undefined, undefined, undefined, undefined, undefined, selectedYouyunGpuCount);
  }, [environment, youyunSpecs, youyunLoading, youyunError, refreshYouyunSpecs]);
  useEffect(() => {
    if (environment === "youyun" && !youyunImages.length && !youyunImagesLoading && !youyunImagesError) void refreshYouyunImages();
  }, [environment, refreshYouyunImages, youyunImages.length, youyunImagesError, youyunImagesLoading]);
  const openImagePicker = useCallback(() => {
    setPendingYouyunImageId(selectedYouyunImageId);
    setImagePickerOpen(true);
    if (!youyunImages.length && !youyunImagesLoading) void refreshYouyunImages();
  }, [refreshYouyunImages, selectedYouyunImageId, youyunImages.length, youyunImagesLoading]);

  useEffect(() => {
    const requestId = ++youyunPriceRequestRef.current;
    setYouyunResource(null);
    setYouyunPriceError("");
    if (environment !== "youyun" || !youyunSpecs || youyunLoading || youyunError) {
      setYouyunPriceLoading(false);
      return;
    }
    const gpuType = selectedYouyunGpuType || youyunSpecs.gpuType;
    const cpuCores = selectedYouyunCpuCores;
    const memoryGb = selectedYouyunMemoryGb;
    const gpuCount = 1;
    const strategy = "single";
    if (!gpuType || !cpuCores || !memoryGb || !youyunSpecs.capacity.some((item) => item.gpu === gpuCount && item.available && item.cpu === cpuCores && item.memoryGb === memoryGb)) {
      setYouyunPriceLoading(false);
      return;
    }
    let cancelled = false;
    setYouyunPriceLoading(true);
    setYouyunPriceError("");
    void apiRequest<YouyunResource>("/youyun/price", {
      method: "POST",
      body: JSON.stringify({ mode: "manual", gpuType, gpuCount: 1, distributedStrategy: "single", cpuCores, memoryGb, region: selectedYouyunRegion || youyunSpecs.region, zone: selectedYouyunZone || youyunSpecs.zone, imageId: selectedYouyunImageId, instanceMode: selectedYouyunMode })
    }).then((resource) => {
      if (!cancelled && requestId === youyunPriceRequestRef.current) setYouyunResource(resource);
    }).catch((error) => {
      if (!cancelled && requestId === youyunPriceRequestRef.current) {
        setYouyunResource(null);
        setYouyunPriceError(friendlyYouyunError(error instanceof Error ? error.message : "", "price"));
      }
    }).finally(() => {
      if (!cancelled && requestId === youyunPriceRequestRef.current) setYouyunPriceLoading(false);
    });
    return () => {
      cancelled = true;
      if (requestId === youyunPriceRequestRef.current) setYouyunPriceLoading(false);
    };
  }, [environment, youyunSpecs, youyunLoading, youyunError, selectedYouyunGpuType, selectedYouyunCpuCores, selectedYouyunMemoryGb, selectedYouyunImageId, selectedYouyunRegion, selectedYouyunZone, selectedYouyunMode]);

  useEffect(() => {
    if (environment === "youyun") setFormError("");
  }, [environment, selectedYouyunGpuType, selectedYouyunCpuCores, selectedYouyunMemoryGb, selectedYouyunImageId, selectedYouyunRegion, selectedYouyunZone, selectedYouyunMode]);

  useEffect(() => () => {
    ++youyunSpecsRequestRef.current;
    ++youyunPriceRequestRef.current;
  }, []);

  const youyunHasStock = (youyunSpecs?.capacity ?? []).some((spec) => spec.available && spec.gpu === 1);
  const youyunLocations = youyunSpecs?.locations ?? [];
  const selectedYouyunLocation = youyunLocations.find((item) => item.zone === (selectedYouyunZone || youyunSpecs?.locationZone));

  const refreshTasks = useCallback(async () => {
    try {
      const params = new URLSearchParams({ page: String(taskPage), pageSize: "5" });
      if (query.trim()) params.set("query", query.trim());
      if (stateFilter !== "all") params.set("state", stateFilter);
      const result = await apiRequest<{ items: TrainingTask[]; page: number; pageSize: number; total: number; totalPages: number }>(`/tasks?${params}`);
      setTasks(result.items);
      setTaskPageInfo({ pageSize: result.pageSize, total: result.total, totalPages: result.totalPages });
      setBackendState("online");
    } catch {
      setBackendState("offline");
    }
  }, [taskPage, query, stateFilter]);

  useEffect(() => {
    setTaskPage(1);
  }, [query, stateFilter]);

  useEffect(() => {
    void refreshTasks();
  }, [refreshTasks]);

  useEffect(() => {
    if (!shouldPollTasks) return;
    const timer = window.setInterval(() => void refreshTasks(), 2000);
    return () => window.clearInterval(timer);
  }, [refreshTasks, shouldPollTasks]);

  useEffect(() => {
    if (view !== "detail" || !selectedId) return;
    setDetailLogConnection("connecting");
    let cancelled = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let fallbackTimer: number | undefined;
    let lastLogId = 0;
    let snapshotLogReceived = false;
    let terminal = false;
    const seenLogIds = new Set<number>();
    const loadDetail = async () => {
      try {
        const task = await apiRequest<TrainingTask>(`/tasks/${selectedId}`);
        if (cancelled) return;
        setDetailTask((current) => mergeTrainingDetail(current, task, lastLogId > 0));
        setTasks((items) => items.map((item) => item.id === task.id ? task : item));
        setDetailError("");
      } catch (error) {
        if (!cancelled) setDetailError(error instanceof Error ? error.message : "读取任务详情失败。");
      } finally {
        if (!cancelled) setDetailLoading(false);
      }
    };
    setDetailLoading(true);
    void loadDetail();
    const startFallback = () => {
      if (fallbackTimer !== undefined || cancelled) return;
      setDetailLogConnection("fallback");
      fallbackTimer = window.setInterval(() => void loadDetail(), 2000);
    };
    const stopFallback = () => {
      if (fallbackTimer !== undefined) window.clearInterval(fallbackTimer);
      fallbackTimer = undefined;
    };
    const connect = () => {
      if (cancelled) return;
      const suffix = lastLogId > 0 ? `?afterId=${lastLogId}` : "";
      try {
        socket = new WebSocket(apiWebSocketUrl(`/tasks/${selectedId}/stream${suffix}`));
      } catch {
        startFallback();
        reconnectTimer = window.setTimeout(connect, 3000);
        return;
      }
      socket.onopen = () => {
        stopFallback();
        setDetailLogConnection("live");
      };
      socket.onmessage = (event) => {
        let payload: any;
        try { payload = JSON.parse(event.data); } catch { return; }
        if (payload.type === "log") {
          const id = Number(payload.id ?? payload.seq ?? 0);
          if (id > 0 && seenLogIds.has(id)) return;
          if (id > 0) { seenLogIds.add(id); lastLogId = Math.max(lastLogId, id); }
          const message = normalizeInferenceLogs({
            items: [{ createdAt: payload.createdAt, level: payload.level, source: payload.source, message: payload.message }],
          })[0] ?? String(payload.message ?? "");
          if (!message) return;
          const isSnapshotStart = Boolean(payload.snapshot) && !snapshotLogReceived;
          if (isSnapshotStart) snapshotLogReceived = true;
          setDetailTask((current) => {
            if (!current || current.id !== selectedId) return current;
            const logs = current.logs ?? [];
            // The first socket batch is a durable snapshot. Replace the
            // detail response's copy so reconnects never duplicate lines.
            const nextLogs = isSnapshotStart ? [message] : [...logs, message];
            return { ...current, logs: nextLogs };
          });
          return;
        }
        if (payload.type === "state") {
          const task = payload.task;
          setDetailTask((current) => {
            if (!current || current.id !== selectedId) return current;
            return {
              ...current,
              ...(task && typeof task === "object" ? task : {}),
              ...(payload.snapshot && !snapshotLogReceived ? { logs: [] } : {}),
              state: payload.state ?? current.state,
              progress: payload.progress ?? current.progress,
              currentStep: payload.currentStep ?? current.currentStep,
            };
          });
          if (payload.state === "completed" || payload.state === "failed" || payload.state === "stopped") {
            void loadDetail();
          }
          return;
        }
        if (payload.type === "metrics" && Array.isArray(payload.metrics)) {
          setDetailTask((current) => current && current.id === selectedId ? { ...current, lossHistory: payload.metrics } : current);
          return;
        }
        if (payload.type === "terminal") {
          terminal = true;
          void loadDetail();
        }
      };
      socket.onerror = () => {
        startFallback();
      };
      socket.onclose = () => {
        if (cancelled || terminal) return;
        startFallback();
        reconnectTimer = window.setTimeout(connect, 2500);
      };
    };
    connect();
    return () => {
      cancelled = true;
      stopFallback();
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      socket?.close();
      socket = null;
    };
  }, [view, selectedId]);

  useEffect(() => {
    if (view !== "create" || datasetsLoaded || datasetsLoading) return;
    if (executionMode === "cli" || selectedModel) void onLoadDatasets();
  }, [datasetsLoaded, datasetsLoading, executionMode, onLoadDatasets, selectedModel, view]);

  useEffect(() => {
    if (view !== "create" || !modelDropdownOpen) return;
    const needle = modelQuery.trim();
    if (!needle) {
      setModelSearchError("");
      let cancelled = false;
      const source = selectedTrainingModelCatalog;
      if (source === "modelscope" && !localRuntimeEnabled) { setModelOptions([]); setModelSearching(false); return; }
      void apiRequest<ModelSearchResult[]>(`/models/search?q=&limit=${source === "public" ? 50 : 20}&source=${source}`).then((results) => {
        if (cancelled) return;
        const options = results.map((model) => ({
          id: model.id,
          label: model.label,
          size: model.size || (source === "public" ? "优云公共模型库" : "本地已缓存"),
          cached: true,
          publicModelPath: model.publicModelPath,
          source: model.source,
          tag: model.tag,
        }));
        setModelOptions(options);
      }).catch(() => {
        if (!cancelled) setModelOptions([]);
      });
      return () => { cancelled = true; };
    }
    if (needle === selectedModel) return;
    if (needle.length < 2) { setModelOptions([]); return; }
    const timer = window.setTimeout(() => {
      setModelSearching(true);
      setModelSearchError("");
      const source = selectedTrainingModelCatalog;
      if (source === "modelscope" && !localRuntimeEnabled) { setModelOptions([]); setModelSearching(false); return; }
      void apiRequest<ModelSearchResult[]>(`/models/search?q=${encodeURIComponent(needle)}&limit=${source === "public" ? 50 : 20}&source=${source}`).then((results) => {
        const options = results.map((model) => ({
          id: model.id,
          label: model.label,
          size: model.size || (source === "public" ? "优云公共模型库" : `${formatModelBytes(model.fileSize)} · 下载 ${model.downloads.toLocaleString()} 次`),
          cached: model.cached,
          publicModelPath: model.publicModelPath,
          source: model.source,
          tag: model.tag,
        }));
        setModelOptions(options);
      }).catch((error) => {
        setModelSearchError(error instanceof Error ? error.message : source === "public" ? "优云公共模型库搜索失败。" : "ModelScope 搜索失败。");
      }).finally(() => setModelSearching(false));
    }, 450);
    return () => window.clearTimeout(timer);
  }, [localRuntimeEnabled, modelDropdownOpen, modelQuery, selectedModel, selectedTrainingModelCatalog, view]);

  function openDetail(task: TrainingTask) {
    setSelectedId(task.id);
    setDetailTask(null);
    setDetailLoading(true);
    setDetailError("");
    setView("detail");
  }

  function closeDetail() {
    setView("list");
    setDetailTask(null);
    setDetailError("");
  }

  async function handleAction(task: TrainingTask, action: string) {
    if (!localRuntimeEnabled && task.provider !== "youyun" && ["重试", "克隆"].includes(action)) { setDetailError(LOCAL_RUNTIME_DISABLED); return; }
    if (action === "详细") {
      openDetail(task);
      return;
    }
    try {
      let updatedTask: TrainingTask | null = null;
      if (action === "停止" && !window.confirm("确认结束训练吗？系统会请求停止当前进程并将任务标记为已停止；只有训练器已经写出的有效检查点才会作为产物保留。选择取消则继续训练。")) {
        return;
      }
      if (action === "停止") updatedTask = await apiRequest<TrainingTask>(`/tasks/${task.id}/stop`, { method: "POST" });
      if (action === "结束任务并释放资源") {
        if (!window.confirm("确认结束任务并释放优云 GPU 吗？释放后不能继续在该实例上调试。")) return;
        updatedTask = await apiRequest<TrainingTask>(`/tasks/${task.id}/youyun/release`, { method: "POST" });
      }
      if (action === "销毁实例") {
        if (!window.confirm("确认销毁优云实例吗？该操作会立即释放远程 GPU。")) return;
        updatedTask = await apiRequest<TrainingTask>(`/tasks/${task.id}/youyun/destroy`, { method: "POST" });
      }
      if (action === "重试") updatedTask = await apiRequest<TrainingTask>(`/tasks/${task.id}/retry`, { method: "POST" });
      if (action === "克隆") await apiRequest(`/tasks/${task.id}/clone`, { method: "POST" });
      if (action === "删除") await apiRequest(`/tasks/${task.id}`, { method: "DELETE" });
      if (updatedTask) {
        setTasks((items) => items.map((item) => item.id === updatedTask?.id ? updatedTask : item));
        if (selectedId === updatedTask.id) setDetailTask(updatedTask);
      }
      await refreshTasks();
      if (updatedTask && selectedId === updatedTask.id) {
        const currentTask = await apiRequest<TrainingTask>(`/tasks/${updatedTask.id}`);
        setDetailTask(currentTask);
        setTasks((items) => items.map((item) => item.id === currentTask.id ? currentTask : item));
      }
      if (action === "删除" && selectedId === task.id) setView("list");
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "任务操作失败。");
      if (selectedId === task.id) {
        try {
          const currentTask = await apiRequest<TrainingTask>(`/tasks/${task.id}`);
          setDetailTask(currentTask);
          setTasks((items) => items.map((item) => item.id === currentTask.id ? currentTask : item));
        } catch {
          // Preserve the action error; the detail poll will retry automatically.
        }
      }
    }
  }

  function selectModel(modelId: string, publicModelPath?: string) {
    setSelectedModel(modelId);
    setSelectedPublicModelPath(publicModelPath || "");
    setModelQuery(modelId);
    setModelDropdownOpen(false);
    setFormError("");
  }

  function changeTuningMode(mode: TuningMode) {
    setTuningMode(mode);
    setBaselineMessages([]);
    setFormError("");
  }

  function changeTaskType(next: TrainingTaskType) {
    setTaskType(next);
    setSelectedDatasetId("");
    setFormError("");
  }

  function buildTaskPayload() {
    return {
      name: taskName.trim(),
      framework: "llamafactory",
      executionMode: (environment === "youyun" ? "remote" : "local") as BackendExecutionMode,
      provider: environment,
      model: selectedModel,
      taskType,
      tuningMode,
      modelSource: selectedPublicModelPath ? "public" : "modelscope",
      datasetId: selectedDataset?.id ?? "",
      datasetName: selectedDataset?.name ?? "",
      owner: "平台管理员",
      mergeModel,
      validationSplit,
      earlyStopping,
      patience,
      minDelta,
      outputName: outputName.trim(),
      checkpointLimit,
      checkpointStrategy,
      checkpointInterval,
      maxSteps,
      parameters,
      youyunResource: environment === "youyun" ? {
        mode: "manual",
        gpuType: selectedYouyunGpuType || youyunSpecs?.gpuType,
        gpuCount: 1,
        distributedStrategy: "single",
        cpuCores: selectedYouyunCpuCores,
        memoryGb: selectedYouyunMemoryGb,
        region: selectedYouyunRegion || youyunSpecs?.region,
        zone: selectedYouyunZone || youyunSpecs?.zone,
        instanceMode: selectedYouyunMode,
        imageId: selectedYouyunImageId,
        imageName: youyunImageLabel(selectedYouyunImageId, youyunImages),
        imageVersion: youyunImages.find((item) => item.id === selectedYouyunImageId)?.tags?.[0],
        publicModelPath: selectedPublicModelPath || undefined,
      } : undefined,
      baselineMessages,
      chatParameters
    };
  }

  function quoteCli(value: string) {
    return `"${value.replace(/"/g, '\\"')}` + `"`;
  }

  function buildCliCommand() {
    const lines = [
      "python scripts/train_cli.py submit",
      `--name ${quoteCli(taskName || "新建微调任务")}`,
      `--model ${quoteCli(selectedModel)}`,
      `--task-type ${taskType}`,
      `--dataset-id ${quoteCli(selectedDataset?.id ?? "dataset-id")}`,
      `--dataset-name ${quoteCli(selectedDataset?.name ?? "数据集")}`,
      `--output-name ${quoteCli(outputName || "adapter-output")}`,
      `--tuning-mode ${tuningMode}`,
      `--validation-split ${validationSplit}`,
      `--epochs ${parameters.epochs}`,
      `--learning-rate ${parameters.learningRate}`,
      `--batch-size ${parameters.batchSize}`,
      `--gradient-accumulation ${parameters.gradientAccumulation}`,
      `--validation-steps ${parameters.validationSteps}`,
      `--max-sequence-length ${parameters.maxSequenceLength}`,
      `--lora-rank ${parameters.loraRank}`,
      `--lora-alpha ${parameters.loraAlpha}`,
      `--lora-dropout ${parameters.loraDropout}`,
      `--lr-scheduler ${parameters.lrScheduler}`,
      ...(parameters.loraPlusEnabled ? [`--lora-plus-ratio ${parameters.loraPlusRatio}`] : []),
      `--target-modules ${parameters.targetModules}`,
      `--optimizer ${parameters.optimizer}`,
      `--warmup-ratio ${parameters.warmupRatio}`,
      `--weight-decay ${parameters.weightDecay}`,
      `--max-grad-norm ${parameters.maxGradNorm}`,
      `--seed ${parameters.seed}`,
      `--logging-steps ${parameters.loggingSteps}`,
      parameters.packing ? "--packing" : "--no-packing",
      parameters.gradientCheckpointing ? "--gradient-checkpointing" : "--no-gradient-checkpointing",
      `--checkpoint-limit ${checkpointLimit}`,
      `--checkpoint-strategy ${checkpointStrategy}`,
      `--checkpoint-interval ${checkpointInterval}`,
      `--max-steps ${maxSteps}`
    ];
    if (earlyStopping) lines.push("--early-stopping", `--patience ${patience}`, `--min-delta ${minDelta}`);
    if (mergeModel) lines.push("--merge-model");
    return lines.join("\n");
  }

  function updateCliDatasetLines(command: string, dataset: Dataset) {
    const replacements = new Map([
      ["--task-type", `--task-type ${dataset.structure ?? "SFT"}`],
      ["--dataset-id", `--dataset-id ${quoteCli(dataset.id)}`],
      ["--dataset-name", `--dataset-name ${quoteCli(dataset.name)}`]
    ]);
    const found = new Set<string>();
    const lines = command.split("\n").map((line) => {
      const option = [...replacements.keys()].find((key) => new RegExp(`^\\s*${key}(?:\\s|$)`).test(line));
      if (!option) return line;
      found.add(option);
      return replacements.get(option) ?? line;
    });
    for (const [option, line] of replacements) {
      if (!found.has(option)) lines.push(line);
    }
    return lines.join("\n");
  }

  function selectCliDataset(datasetId: string) {
    const dataset = publishedTextDatasets.find((item) => item.id === datasetId);
    if (!dataset) return;
    setTaskType(dataset.structure ?? "SFT");
    setSelectedDatasetId(dataset.id);
    setCliCommandText((command) => updateCliDatasetLines(command || buildCliCommand(), dataset));
    setCliNotice(`已写入数据集：${dataset.name}`);
    setFormError("");
  }

  async function copySelectedDatasetId() {
    if (!selectedDataset) return;
    try {
      await navigator.clipboard.writeText(selectedDataset.id);
      setCliNotice(`数据集 ID 已复制：${selectedDataset.id}`);
    } catch {
      setCliNotice("复制失败，请手动复制数据集 ID");
    }
  }

  async function copyCliCommand() {
    try {
      await navigator.clipboard.writeText(cliCommandText || buildCliCommand());
      setCliNotice("命令已复制");
    } catch {
      setCliNotice("复制失败，请手动复制命令");
    }
  }

  async function retryArtifactSync(taskId: string) {
    try {
      const task = await apiRequest<TrainingTask>(`/tasks/${taskId}/artifacts/sync`, { method: "POST" });
      setDetailTask(task);
      setTasks((items) => items.map((item) => item.id === task.id ? task : item));
      setDetailError("");
    } catch (error) {
      setDetailError(error instanceof Error ? error.message : "重新同步训练产物失败。");
    }
  }

  async function createTask() {
    if (executionMode === "cli") {
      const command = (cliCommandText || buildCliCommand()).trim();
      if (!command) {
        setFormError("请输入训练命令。");
        return;
      }
      setSubmitting(true);
      try {
        const task = await apiRequest<TrainingTask>("/tasks/cli", {
          method: "POST",
          body: JSON.stringify({ command })
        });
        setTasks((items) => [task, ...items]);
        setSelectedId(task.id);
        setView("detail");
        setCliNotice("");
        setFormError("");
      } catch (error) {
        setFormError(error instanceof Error ? error.message : "命令行任务校验失败。");
      } finally {
        setSubmitting(false);
      }
      return;
    }
    if (!taskName.trim()) {
      setFormError("请填写任务名称。");
      return;
    }
    if (!selectedModel) {
      setFormError("请先选择基础模型。");
      return;
    }
    if (environment === "youyun") {
      if (youyunLoading) {
        setFormError("正在查询当前地域的 GPU 与 CPU / 内存库存，请稍后再提交。");
        return;
      }
      if (selectedYouyunImageContainer === false && selectedPublicModelPath) {
        setFormError("虚机实例不支持公共模型库模型，请重新选择 ModelScope 基础模型。");
        return;
      }
      if (youyunImagesError || youyunImagesLoading || !selectedYouyunImageId) {
        setFormError(youyunImagesError || "优云官方镜像尚未加载完成，无法提交远程任务，请刷新后重试。");
        return;
      }
      if (youyunError) {
        setFormError(`优云 GPU 不可用：${youyunError}`);
        return;
      }
      if (!youyunHasStock) {
        setFormError("优云 GPU 当前无可用库存，无法提交远程任务，请稍后重试。");
        return;
      }
      if (!selectedYouyunCpuCores || !selectedYouyunMemoryGb) {
        setFormError("请选择有库存的单卡 CPU 与内存规格。");
        return;
      }
      if (youyunPriceLoading || !youyunResource) {
        setFormError(youyunPriceError || "正在查询优云价格，请稍后再提交。");
        return;
      }
    }
    if (!selectedDataset) {
      setFormError(`请先在数据集管理中发布一个文本生成 ${taskType} 数据集。`);
      return;
    }
    if (!outputName.trim()) {
      setFormError("请填写训练产出的模型名称。");
      return;
    }
    setSubmitting(true);
    try {
      const payload = buildTaskPayload();
      const task = await apiRequest<TrainingTask>("/tasks", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      setTasks((items) => [task, ...items]);
      setSelectedId(task.id);
      setView("detail");
      setTaskName("");
      setBaselineMessages([]);
      setFormError("");
      clearSessionDraft(TRAINING_DRAFT_STORAGE_KEY);
      setBackendState("online");
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "提交训练任务失败。");
      setBackendState((current) => backendStateAfterSubmitError(current, error));
    } finally {
      setSubmitting(false);
    }
  }

  if (view === "baseline-chat") {
    return <BaselineChatWorkspace model={selectedModel} tuningMode={tuningMode} datasetId={selectedDataset?.id ?? ""} messages={baselineMessages} parameters={chatParameters} onMessagesChange={setBaselineMessages} onParametersChange={setChatParameters} onBack={() => setView("create")} />;
  }

  if (view === "compare" && selectedTask) {
    return <ComparisonWorkspace task={selectedTask} onBack={() => setView("detail")} />;
  }

  if (view === "create") {
    return (
      <section className="content tuning-content">
        <div className="title-row">
          <div><BackButton label="返回微调任务" onClick={requestCreateBack} /><h1>新建微调任务</h1><div className="subtitle">创建文本生成 {taskType} LoRA / QLoRA 任务，{environment === "youyun" ? "远程优云 GPU 一次性实例执行（完成或停止后可对比，随后手动释放）" : "由 LLaMA-Factory 本地训练执行器执行"}</div></div>
        </div>
        <div className="tuning-create-layout">
          <div className="tuning-form-stack">
            <section id="training-section-basic" className="panel tuning-section">
              <div className="section-heading"><div><h2>基本信息</h2><p>选择 LoRA 16-bit 或 QLoRA 4-bit 进行对比训练。</p></div><span className="section-index">01</span></div>
              {executionMode === "web" && <div className="field"><label className="field-label">任务名称 <span className="required">*</span></label><SuggestedNameInput value={taskName} suggestion={taskNameSuggestion} onChange={(value) => { setTaskName(value); setFormError(""); }} /></div>}
              <div className="field"><label className="field-label">执行方式</label><div className="segmented execution-mode-segmented"><button type="button" className={executionMode === "web" ? "selected" : ""} onClick={() => { setExecutionMode("web"); setCliNotice(""); }}><strong>网页提交</strong><small>进入所选环境的训练队列</small></button>{localRuntimeEnabled && <button type="button" className={executionMode === "cli" ? "selected" : ""} onClick={() => { setExecutionMode("cli"); setCliCommandText(buildCliCommand()); setCliNotice(""); }}><strong>命令行执行</strong><small>编辑命令后直接提交</small></button>}</div></div>
              {executionMode === "web" && <div className="field"><label className="field-label">执行环境</label><div className="segmented execution-mode-segmented">{localRuntimeEnabled && <button type="button" className={environment === "local" ? "selected" : ""} onClick={() => { setEnvironment("local"); setFormError(""); }}><strong>本地 CPU 服务器</strong><small>现有本地训练链路</small></button>}<button type="button" className={environment === "youyun" ? "selected" : ""} onClick={() => { setEnvironment("youyun"); setFormError(""); }}><strong>远程优云 GPU</strong><small>一次性实例 · 按量计费</small></button></div></div>}
              {executionMode === "web" && environment === "youyun" && <div className="youyun-specs-panel">
                <div className="youyun-specs-head">
                  <div><strong>优云 GPU 配置与实时库存</strong><small>{youyunSpecs ? `${youyunRegionLabel(youyunSpecs.region)} · ${youyunSpecs.chargeType === "Postpay" ? "按量计费" : youyunSpecs.chargeType}` : "正在查询可用规格..."}</small></div>
                   <button type="button" className="secondary-btn compact-btn" disabled={youyunLoading} onClick={() => void refreshYouyunSpecs(undefined, undefined, undefined, undefined, undefined, undefined, true)}>{youyunLoading ? "查询中..." : "刷新库存"}</button>
                </div>
                {youyunError && <div className="youyun-specs-error">{youyunError}</div>}
                <div className="youyun-image-picker-row"><div><span className="youyun-capacity-label">执行镜像</span><strong>{selectedYouyunImageId ? youyunImageLabel(selectedYouyunImageId, youyunImages) : "尚未选择"}</strong><small>{selectedYouyunImageId ? `镜像 ID：${selectedYouyunImageId} · ${youyunImageRuntimeLabel(selectedYouyunImageContainer)}` : "打开选择器后选择训练镜像"}</small></div><button type="button" className="secondary-btn compact-btn" disabled={youyunImagesLoading} onClick={openImagePicker}>{youyunImagesLoading ? "加载中..." : "更换镜像"}</button></div>
                {youyunSpecs?.zones?.length ? <div className="youyun-location-picker">
                  <span className="field-label">地域 / 可用区</span>
                  <div className="youyun-region-list">{youyunLocations.map((item) => <button type="button" key={item.zone} disabled={item.supported === false} title={item.disabledReason || item.describe || item.zone} className={`youyun-region-option${(selectedYouyunZone || youyunSpecs.locationZone || youyunSpecs.zone) === item.zone ? " selected" : ""}${item.supported === false ? " disabled" : ""}`} onClick={() => {
                    if (item.supported === false) return;
                    const nextMode = youyunPreferredMode(item, selectedYouyunImageContainer, selectedYouyunMode);
                    setSelectedYouyunRegion(item.region);
                    setSelectedYouyunZone(item.zone);
                    setSelectedYouyunMode(nextMode);
                    setYouyunResource(null);
                    setSelectedYouyunGpuType("");
                     setSelectedYouyunCpuCores(null);
                     setSelectedYouyunMemoryGb(null);
                    setYouyunPriceError("");
                     void refreshYouyunSpecs("", item.region, item.zone, selectedYouyunImageId, nextMode);
                  }}>{item.displayName || item.describe || item.zone}</button>)}</div>
                  <small>每个地域对应优云的精确可用区；GPU 是否可用以当前 Zone 的实时库存为准。</small>
                </div> : null}
                {youyunSpecs && <>
                  <div className="youyun-selection-summary"><strong>手动选择单卡 GPU 与 CPU / 内存规格</strong><small>当前地域：{selectedYouyunLocation?.displayName || youyunRegionLabel(youyunSpecs.region)} · 可用区 {youyunSpecs.zone}；当前版本暂支持单卡训练。价格会随所选规格实时刷新。</small></div>
                  <div className="youyun-machine-list">
                    {Array.from(new Map(youyunSpecs.machineTypes.map((machine) => [`${machine.name}-${machine.graphicsMemoryGb}`, machine])).values()).map((machine) => (
                      <button type="button" key={`${machine.name}-${machine.graphicsMemoryGb}`} className={`youyun-machine-item${youyunMachineIsUnavailable(machine, "training") ? " sold-out" : ""}${selectedYouyunGpuType === machine.name ? " selected" : ""}`} disabled={youyunLoading || youyunMachineIsUnavailable(machine, "training")} onClick={() => { setSelectedYouyunGpuType(machine.name); setSelectedYouyunCpuCores(null); setSelectedYouyunMemoryGb(null); if (machine.zone && machine.zone !== selectedYouyunZone) setSelectedYouyunZone(selectedYouyunZone); setYouyunSpecs((current) => current ? { ...current, gpuType: machine.name, gpuMemoryGb: machine.graphicsMemoryGb, capacity: [] } : current); setYouyunResource(null); setYouyunPriceError(""); void refreshYouyunCapacity(machine.name, selectedYouyunRegion || youyunSpecs.region, selectedYouyunZone, 1); }}>
                        <strong>{machine.graphicsMemoryGb ? `${machine.graphicsMemoryGb}G` : "GPU"} {machine.name}</strong>
                        <small>{machine.graphicsMemoryGb ? `${machine.graphicsMemoryGb}GB 显存 · ` : ""}{machine.description || machine.zone}{machine.supportReason ? ` · ${machine.supportReason}` : ""}{selectedYouyunGpuType === machine.name && youyunSpecs.capacity.find((spec) => spec.gpu === 1 && spec.available) ? ` · 默认 ${youyunSpecs.capacity.find((spec) => spec.gpu === 1 && spec.available)!.cpu} 核 / ${youyunSpecs.capacity.find((spec) => spec.gpu === 1 && spec.available)!.memoryGb}GB 内存` : ""}</small>
                        <span className={youyunMachineInventoryClass(machine)}>{youyunMachineInventoryLabel(machine)}</span>
                        <span className={machine.trainingSupported === false ? "tag-off" : "tag-ready"}>{machine.trainingSupported === false ? "暂不支持训练" : "支持训练"}</span>
                      </button>
                    ))}
                  </div>
                  <div className="youyun-capacity-list youyun-single-capacity-list">
                    {youyunSpecs.capacity.filter((spec) => spec.gpu === 1).map((spec) => (
                      <button type="button" key={`${spec.gpu}-${spec.cpu}-${spec.memoryGb}`} disabled={youyunLoading || !spec.available} className={`youyun-capacity-item${selectedYouyunCpuCores === spec.cpu && selectedYouyunMemoryGb === spec.memoryGb ? " selected" : ""}`} onClick={() => { setSelectedYouyunCpuCores(spec.cpu); setSelectedYouyunMemoryGb(spec.memoryGb); }}>
                        <span>{spec.cpu} 核 / {spec.memoryGb}GB 内存</span>
                        <span className={spec.available ? "tag-ready" : "tag-off"}>{spec.available ? "可选" : "CPU / 内存无库存"}</span>
                      </button>
                    ))}
                  </div>
                   <div className="youyun-price-summary"><div><strong>预计按量费用</strong><span>{youyunPriceLoading ? "查询中..." : youyunResource ? `${youyunResource.price.currency === "CNY" ? "¥" : `${youyunResource.price.currency} `}${youyunResource.price.total.toFixed(2)} / 小时` : "暂不可获取"}</span></div>{youyunResource && <small>实例 ¥{youyunResource.price.instance.toFixed(2)} · 系统盘 ¥{youyunResource.price.systemDisks.toFixed(2)} · 数据盘 ¥{youyunResource.price.disks.toFixed(2)} · 镜像 ¥{youyunResource.price.image.toFixed(2)}；实际以优云账单为准</small>}{youyunPriceError && <div className="youyun-specs-error">{youyunPriceError}<button type="button" className="link-btn" onClick={() => void refreshYouyunSpecs(undefined, undefined, undefined, undefined, undefined, undefined, true)}>重新查询</button></div>}</div>
                  {youyunLoading && <div role="status">正在更新当前地域的 GPU 与 CPU / 内存库存...</div>}
                  {!youyunLoading && !youyunError && youyunInventoryMessage(youyunSpecs, "training") && <div className="youyun-specs-error">{youyunInventoryMessage(youyunSpecs, "training")}</div>}
                </>}
              </div>}
              {executionMode === "cli" && <div className="cli-config-panel"><div className="cli-config-head"><div><strong>命令行任务配置</strong><span>一行一个参数，可留空行；仅允许训练参数，提交前由后端校验</span></div><div className="cli-config-actions"><button type="button" className="secondary-btn compact-btn" onClick={() => void copyCliCommand()}>复制配置</button></div></div>{!datasetsLoaded || datasetsLoading ? <div className="dataset-empty cli-dataset-empty"><div><strong>正在读取已发布数据集</strong><span>仅在命令行配置需要绑定数据集时加载。</span></div></div> : datasetsError ? <div className="dataset-empty cli-dataset-empty"><div><strong>数据集读取失败</strong><span>{datasetsError}</span></div><button type="button" className="secondary-btn" onClick={() => void onLoadDatasets()}>重新读取</button></div> : publishedTextDatasets.length ? <div className="cli-dataset-binding"><label htmlFor="cli-dataset-select">训练数据集</label><div className="cli-dataset-controls"><select id="cli-dataset-select" value={selectedDataset?.id ?? ""} onChange={(event) => selectCliDataset(event.target.value)}>{publishedTextDatasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name} · {dataset.id}</option>)}</select><button type="button" className="secondary-btn compact-btn" onClick={() => void copySelectedDatasetId()}>复制 ID</button></div><div className="cli-dataset-current"><span>当前写入命令</span><code>{selectedDataset?.id}</code><small>{selectedDataset?.structure ?? "SFT"} · 已发布</small></div></div> : <div className="dataset-empty cli-dataset-empty"><div><strong>暂无已发布的文本数据集</strong><span>请先发布数据集，再返回生成命令配置。</span></div><button type="button" className="secondary-btn" onClick={onOpenDatasets}>前往数据集管理</button></div>}<textarea className="cli-config-editor" value={cliCommandText || buildCliCommand()} onChange={(event) => setCliCommandText(event.target.value)} spellCheck={false} />{cliNotice && <div className="artifact-message">{cliNotice}</div>}</div>}
              {executionMode === "web" && <><div className="field"><label className="field-label">训练类型 <span className="required">*</span></label><div className="segmented tuning-mode-segmented"><button type="button" className={taskType === "SFT" ? "selected" : ""} onClick={() => changeTaskType("SFT")}><strong>SFT 指令微调</strong><small>messages 对话数据</small></button><button type="button" className={taskType === "DPO" ? "selected" : ""} onClick={() => changeTaskType("DPO")}><strong>DPO 偏好训练</strong><small>chosen / rejected 数据</small></button></div></div><div className="field"><label className="field-label">微调方式 <span className="required">*</span></label><div className="segmented tuning-mode-segmented"><button type="button" className={tuningMode === "lora" ? "selected" : ""} onClick={() => changeTuningMode("lora")}><strong>LoRA</strong><small>16-bit 精度</small></button><button type="button" className={tuningMode === "qlora" ? "selected" : ""} onClick={() => changeTuningMode("qlora")}><strong>QLoRA</strong><small>4-bit 精度</small></button></div></div><div className="fixed-option-row"><div><span className="fixed-option-label">任务类型</span><strong>文本生成 · {taskType}</strong></div><div><span className="fixed-option-label">基模精度</span><strong>{tuningMode === "lora" ? "LoRA · 16-bit" : "QLoRA · 4-bit"}</strong></div><div><span className="fixed-option-label">训练框架</span><strong>LLaMA-Factory</strong></div></div></>}
            </section>

            <section id="training-section-model-data" className={`panel tuning-section ${executionMode === "cli" ? "cli-secondary-section" : ""}`}>
              <div className="section-heading"><div><h2>模型与数据</h2><p>{usesYouyunPublicModels ? "直接搜索优云容器公共模型库，选中后使用容器内路径。" : "优云远程下载兜底：填写准确的 ModelScope / Hugging Face Repo ID。"}</p></div><span className="section-index">02</span></div>
              <div className="field"><label className="field-label">基础模型 <span className="required">*</span></label>
                <div className="model-combobox" onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setModelDropdownOpen(false); }}>
                  <input className="input" value={modelQuery} aria-label={usesYouyunPublicModels ? "搜索优云公共模型库" : "搜索模型或 Repo ID"} placeholder={usesYouyunPublicModels ? "输入关键词搜索优云公共模型库" : "点击选择，或输入关键词 / Repo ID 检索"} onFocus={() => setModelDropdownOpen(true)} onClick={() => setModelDropdownOpen(true)} onChange={(event) => { setModelQuery(event.target.value); setModelDropdownOpen(true); }} />
                  {modelDropdownOpen && <div className="model-dropdown">
                    {modelOptions.map((model) => {
                      return <button type="button" key={model.id} className={`model-option ${selectedModel === model.id ? "selected" : ""}`} onClick={() => selectModel(model.id, model.publicModelPath)}>
                        <span className="model-mark">Q</span><span className="model-copy"><strong>{model.label}</strong><small>{model.id} · {model.size}</small></span>
                        <span className={`model-state model-state-${model.cached ? "ready" : "not-downloaded"}`}>{model.source === "youyun-public" ? "公共模型" : model.cached ? "本地已有" : "训练时下载"}</span>
                      </button>;
                    })}
                    {directRepoId && <button type="button" className="model-option" onClick={() => selectModel(directRepoId)}><span className="model-mark">ID</span><span className="model-copy"><strong>优云远程下载兜底（Repo ID）</strong><small>{directRepoId} · 仅在优云实例上从 ModelScope / Hugging Face 下载</small></span><span className="model-state model-state-not-downloaded">训练时下载</span></button>}
                    {modelSearching && <div className="row-sub">正在搜索{usesYouyunPublicModels ? "优云公共模型库" : " ModelScope"}...</div>}
                    {modelSearchError && <div className="form-error">{modelSearchError}</div>}
                    {!modelSearching && !modelSearchError && !modelOptions.length && !directRepoId && <div className="empty">{modelQuery.trim() ? "没有匹配的文本生成模型" : usesYouyunPublicModels ? "暂无优云公共模型，输入关键词搜索" : "暂无本地缓存模型，输入关键词搜索 ModelScope"}</div>}
                  </div>}
                </div>
                {selectedModel && <div className="model-selected-note">已选择基模：<code>{selectedModel}</code> · {selectedPublicModelPath ? `优云公共模型库：${selectedPublicModelPath}` : modelOptions.find((model) => model.id === selectedModel)?.cached ? "使用本地缓存" : environment === "youyun" ? "优云实例训练启动时从 ModelScope 下载" : "训练任务启动时从 ModelScope 下载"}</div>}
              </div>
              {executionMode === "web" ? (
                selectedModel ? (
                  <div className="field">
                    <label className="field-label">训练数据集 <span className="required">*</span></label>
                    {!datasetsLoaded || datasetsLoading ? <div className="dataset-empty"><div><strong>正在读取已发布数据集</strong><span>选择基模后才会加载可用训练数据。</span></div></div> : datasetsError ? <div className="dataset-empty"><div><strong>数据集读取失败</strong><span>{datasetsError}</span></div><button type="button" className="secondary-btn" onClick={() => void onLoadDatasets()}>重新读取</button></div> : publishedDatasets.length ? <>
                      <select className="input dataset-select" value={selectedDatasetId} onChange={(event) => { setSelectedDatasetId(event.target.value); setFormError(""); }}>
                        <option value="">请选择训练数据集</option>
                        {publishedDatasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name} · {dataset.count} · 已发布</option>)}
                      </select>
                      {selectedDataset && <div className="dataset-selected-note">已选择：{selectedDataset.name} · {taskType === "DPO" ? "preference JSONL" : "messages JSONL"} · {selectedDataset.count}</div>}
                    </> : <div className="dataset-empty"><div><strong>暂无可用的文本 {taskType} 数据集</strong><span>请先在数据集管理中上传、校验并发布数据集。</span></div><button type="button" className="secondary-btn" onClick={onOpenDatasets}>前往数据集管理</button></div>}
                  </div>
                ) : (
                  <div className="field">
                    <label className="field-label">训练数据集 <span className="required">*</span></label>
                    <div className="dataset-empty"><div><strong>请先选择基础模型</strong><span>选择基模后，即可选择训练数据集。</span></div></div>
                  </div>
                )
              ) : (
                <div className="field">
                  <label className="field-label">训练数据集 <span className="required">*</span></label>
                  {!datasetsLoaded || datasetsLoading ? <div className="dataset-empty"><div><strong>正在读取已发布数据集</strong><span>正在为当前任务准备数据集选项。</span></div></div> : datasetsError ? <div className="dataset-empty"><div><strong>数据集读取失败</strong><span>{datasetsError}</span></div><button type="button" className="secondary-btn" onClick={() => void onLoadDatasets()}>重新读取</button></div> : publishedDatasets.length ? <div className="dataset-options">{publishedDatasets.map((dataset) => <button type="button" key={dataset.id} className={`dataset-choice ${selectedDataset?.id === dataset.id ? "selected" : ""}`} onClick={() => { setSelectedDatasetId(dataset.id); setFormError(""); }}><span className="dataset-choice-icon">{taskType}</span><span><strong>{dataset.name}</strong><small>已发布 · {taskType === "DPO" ? "preference JSONL" : "messages JSONL"} · {dataset.count}</small></span><span className="choice-check">{selectedDataset?.id === dataset.id ? "已选择" : "选择"}</span></button>)}</div> : <div className="dataset-empty"><div><strong>暂无可用的文本 {taskType} 数据集</strong><span>请先在数据集管理中上传、校验并发布数据集。</span></div><button type="button" className="secondary-btn" onClick={onOpenDatasets}>前往数据集管理</button></div>}
                </div>
              )}
            </section>

            {selectedDataset ? <>
             <section id="training-section-parameters" className={`panel tuning-section ${executionMode === "cli" ? "cli-secondary-section" : ""}`}>
              <div className="section-heading"><div><h2>训练参数</h2><p>常用参数直接配置，其余保持推荐默认值。</p></div><span className="section-index">03</span></div>
              <div className="parameter-grid">
                <ParameterInput label="单卡批次大小" value={parameters.batchSize} onChange={(value) => setParameters({ ...parameters, batchSize: Number(value) })} />
                <ParameterInput label="梯度累积步数" value={parameters.gradientAccumulation} onChange={(value) => setParameters({ ...parameters, gradientAccumulation: Number(value) })} />
                <ParameterInput label="验证步数" value={parameters.validationSteps} onChange={(value) => setParameters({ ...parameters, validationSteps: Number(value) })} />
                <ParameterInput inputType="text" label="学习率" value={parameters.learningRate} onChange={(value) => setParameters({ ...parameters, learningRate: value })} />
                <ParameterInput label="LoRA Alpha" value={parameters.loraAlpha} onChange={(value) => setParameters({ ...parameters, loraAlpha: Number(value) })} />
                <ParameterInput inputType="decimal" label="LoRA Dropout" value={parameters.loraDropout} onChange={(value) => setParameters({ ...parameters, loraDropout: Number(value) })} />
                <ParameterInput label="LoRA Rank" value={parameters.loraRank} onChange={(value) => setParameters({ ...parameters, loraRank: Number(value) })} />
                <ParameterSelect label="学习率调整策略" value={parameters.lrScheduler} options={[['linear', 'Linear'], ['cosine', 'Cosine'], ['constant', 'Constant']]} onChange={(value) => setParameters({ ...parameters, lrScheduler: value as TrainingParameters['lrScheduler'] })} />
                <ParameterInput label="最大序列长度" value={parameters.maxSequenceLength} onChange={(value) => setParameters({ ...parameters, maxSequenceLength: Number(value) })} />
                <ParameterInput inputType="decimal" label="训练轮数 Epochs" value={parameters.epochs} onChange={(value) => setParameters({ ...parameters, epochs: Number(value) })} />
              </div>
              <div className="effective-batch">有效批次大小：<strong>{parameters.batchSize * parameters.gradientAccumulation}</strong>（单卡批次 × 梯度累积）</div>
              <button className="advanced-toggle" onClick={() => setAdvancedOpen((open) => !open)}><span>{advancedOpen ? "收起" : "展开"}高级设置</span><span>{advancedOpen ? "⌃" : "⌄"}</span></button>
              {advancedOpen && <div className="advanced-panel"><div className="advanced-form-grid">
                <ToggleSetting label="启用 LoRA+" description="让 LoRA 的两组参数使用不同学习率；默认关闭，适合先用普通 LoRA 建立基线" checked={parameters.loraPlusEnabled} onChange={(checked) => setParameters({ ...parameters, loraPlusEnabled: checked })} />
                {parameters.loraPlusEnabled && <ParameterInput inputType="decimal" label="LoRA+ 学习率比例" value={parameters.loraPlusRatio} onChange={(value) => setParameters({ ...parameters, loraPlusRatio: Number(value) })} />}
                <ParameterSelect label="LoRA 作用模块" value={parameters.targetModules} options={[['auto', '自动识别'], ['attention', 'q/k/v/o 投影'], ['all-linear', '全部线性层']]} onChange={(value) => setParameters({ ...parameters, targetModules: value as TrainingParameters['targetModules'] })} />
                <ParameterSelect label="优化器" value={parameters.optimizer} options={[['adamw_torch', 'AdamW Torch（稳定通用）'], ['adamw_8bit', 'AdamW 8-bit（显存优化）'], ['paged_adamw_8bit', 'Paged AdamW 8-bit（显存峰值优化）']]} onChange={(value) => setParameters({ ...parameters, optimizer: value as TrainingParameters['optimizer'] })} />
                <ParameterInput inputType="decimal" label="Warmup Ratio" value={parameters.warmupRatio} onChange={(value) => setParameters({ ...parameters, warmupRatio: Number(value) })} />
                <ParameterInput inputType="decimal" label="Weight Decay" value={parameters.weightDecay} onChange={(value) => setParameters({ ...parameters, weightDecay: Number(value) })} />
                <ParameterInput inputType="decimal" label="Max Grad Norm" value={parameters.maxGradNorm} onChange={(value) => setParameters({ ...parameters, maxGradNorm: Number(value) })} />
                <ParameterInput label="随机种子" value={parameters.seed} onChange={(value) => setParameters({ ...parameters, seed: Number(value) })} />
                <ParameterInput label="日志记录步数" value={parameters.loggingSteps} onChange={(value) => setParameters({ ...parameters, loggingSteps: Number(value) })} />
              </div><div className="advanced-toggles"><ToggleSetting label="样本 Packing" description="将短样本拼接以提高训练吞吐" checked={parameters.packing} onChange={(checked) => setParameters({ ...parameters, packing: checked })} /><ToggleSetting label="梯度检查点" description="降低显存占用，适合优云 GPU 或长序列训练；代价是训练速度略慢" checked={parameters.gradientCheckpointing} onChange={(checked) => setParameters({ ...parameters, gradientCheckpointing: checked })} /></div></div>}
            </section>

             <section id="training-section-validation" className={`panel tuning-section validation-section ${executionMode === "cli" ? "cli-secondary-section" : ""}`}>
              <div className="section-heading"><div><h2>验证与早停</h2><p>验证集可选；未启用验证时，最佳点按训练 Loss 选择。</p></div><span className="section-index">04</span></div>
              <div className="validation-control"><div className="validation-control-head"><label className="field-label">验证集拆分比例</label><strong>{validationSplit}%</strong></div><input type="range" min="0" max="30" step="5" value={validationSplit} onChange={(event) => setValidationSplit(Number(event.target.value))} /><div className="range-labels"><span>0%</span><span>30%</span></div><p>{validationSplit > 0 ? `任务启动时按随机种子 ${parameters.seed} 拆分，原始数据集不会被修改。` : "设置为 0% 表示只训练不验证，最佳点将按训练 Loss 选择。"}</p></div>
              <ToggleSetting label="启用自动早停" description="连续多次 Loss 没有改善后自动结束训练；手动停止始终可用" checked={earlyStopping} onChange={setEarlyStopping} />
              {earlyStopping && <div className="validation-control"><div className="checkpoint-settings"><ParameterInput label="Patience（未改善次数）" value={patience} onChange={(value) => setPatience(Math.max(1, Number(value)))} /><ParameterInput inputType="decimal" label="Min Delta（最小改善）" value={minDelta} onChange={(value) => setMinDelta(Math.max(0, Number(value)))} /></div><p>{validationSplit > 0 ? "自动早停依据验证 Loss 判断。" : "当前未启用验证集，自动早停依据训练 Loss 判断，不能代表泛化效果最佳。"}</p></div>}
            </section>

             <section id="training-section-output" className={`panel tuning-section output-section ${executionMode === "cli" ? "cli-secondary-section" : ""}`}>
              <div className="section-heading"><div><h2>训练产出与检查点</h2><p>配置模型名称、导出内容和训练中间产物。</p></div><span className="section-index">05</span></div>
              <div className="field"><label className="field-label">模型名称 <span className="required">*</span></label><SuggestedNameInput value={outputName} suggestion={outputNameSuggestion} onChange={(value) => { setOutputName(value); setFormError(""); }} /></div>
              <div className="output-choice"><div><strong>LoRA Adapter 权重</strong><small>默认产物，训练完成后登记到模型管理。</small></div><span className="output-default">默认输出</span></div>
              <label className="toggle-row"><span><strong>训练完成后合并模型</strong><small>额外生成可独立部署的完整模型，会占用更多磁盘空间。</small></span><input type="checkbox" checked={mergeModel} onChange={(event) => setMergeModel(event.target.checked)} /><i /></label>
              <div className="checkpoint-settings"><ParameterInput label="最大训练步数（-1 按 Epoch）" value={maxSteps} onChange={(value) => setMaxSteps(Math.max(-1, Number(value)))} /><ParameterInput label="最多保留检查点" value={checkpointLimit} onChange={(value) => setCheckpointLimit(Number(value))} /><div className="parameter-field"><span>检查点保存方式</span><div className="segmented checkpoint-segmented"><button type="button" className={checkpointStrategy === "epoch" ? "selected" : ""} onClick={() => setCheckpointStrategy("epoch")}>每轮保存</button><button type="button" className={checkpointStrategy === "steps" ? "selected" : ""} onClick={() => setCheckpointStrategy("steps")}>按步数</button></div></div>{checkpointStrategy === "steps" && <ParameterInput label="每 N 步保存" value={checkpointInterval} onChange={(value) => setCheckpointInterval(Number(value))} />}</div><p className="field-note">检查点用于保留训练中间结果；当前任务暂不支持从检查点断点续训。</p>
              <p className="field-help">启用验证时，按“验证步数”同步验证并保存检查点，覆盖上方保存周期；至少保留最佳和最新检查点。结束后单独保留最终权重，不用最佳权重覆盖最终权重。</p>
            </section>
            </> : (
               <section id="training-section-parameters" className={`panel tuning-section ${executionMode === "cli" ? "cli-secondary-section" : ""}`}>
                <div className="section-heading"><div><h2>训练参数</h2><p>选择基础模型与训练数据集后，将展示训练参数设置。</p></div><span className="section-index">03</span></div>
              </section>
            )}

            {formError && <div className="form-error">{formError}</div>}
            <div className="training-form-footer"><button className="secondary-btn" disabled={submitting} onClick={requestCreateBack}>取消</button><button className="primary-btn" disabled={submitting || backendState === "offline"} onClick={createTask}>{submitting ? "正在校验并提交" : "提交训练任务"}</button></div>
          </div>
          <aside className="panel training-summary" aria-label="已填写内容导航">
            <div className="training-summary-head">
              <div>
                <div className="summary-eyebrow">TASK FLOW</div>
                <h2>已填写内容</h2>
              </div>
              <BackButton className="training-summary-back" label="返回微调任务" onClick={requestCreateBack} />
            </div>
            <nav className="training-summary-nav">
              {createNavItems.map((item) => (
                <button type="button" className="training-summary-item" key={item.id} onClick={() => jumpToCreateSection(item.id)}>
                  <span className="training-summary-item-index">{item.index}</span>
                  <span className="training-summary-item-copy"><strong>{item.title}</strong><small>{item.summary}</small></span>
                  <span className="training-summary-item-arrow" aria-hidden="true">›</span>
                </button>
              ))}
            </nav>
          </aside>
        </div>
        {imagePickerOpen && <div className="model-modal-layer" role="presentation" onClick={() => setImagePickerOpen(false)}><div className="model-modal youyun-image-modal" role="dialog" aria-modal="true" aria-labelledby="youyun-image-title" onClick={(event) => event.stopPropagation()}><div className="model-modal-head"><div><span className="drawer-kicker">PLATFORM IMAGE</span><h2 id="youyun-image-title">选择训练镜像</h2><p>镜像决定运行环境和兼容的实例类型；支持 Ubuntu/第三方预装镜像。</p></div><button type="button" className="close-btn" onClick={() => setImagePickerOpen(false)} aria-label="关闭镜像选择"><X size={18} /></button></div><div className="youyun-image-tabs"><strong>平台训练镜像</strong></div>{youyunImagesLoading ? <div className="model-empty">正在读取官方训练镜像...</div> : youyunImagesError ? <div className="youyun-specs-error">{youyunImagesError}<button type="button" className="link-btn" onClick={() => void refreshYouyunImages()}>重新读取</button></div> : youyunImages.length ? <div className="youyun-image-grid">{youyunImages.map((image) => <button type="button" key={image.id} className={`youyun-image-card${pendingYouyunImageId === image.id ? " selected" : ""}`} onClick={() => setPendingYouyunImageId(image.id)}><span className="youyun-image-icon">{image.name.toLowerCase().includes("pytorch") ? "PT" : image.name.toLowerCase().includes("cuda") ? "CU" : image.name.toLowerCase().includes("vllm") ? "VL" : "AI"}</span><span className="youyun-image-copy"><strong>{image.name}</strong><small>{image.source || "官方镜像"} · {youyunImageRuntimeLabel(image.container)} · {image.tags.length ? image.tags.join(" / ") : image.type || "训练环境"}</small><em>{image.description || "平台官方训练镜像"}</em></span><span className={pendingYouyunImageId === image.id ? "selected-variant" : "image-ready"}>{pendingYouyunImageId === image.id ? "已选择" : "官方支持"}</span></button>)}</div> : <div className="model-empty">当前没有可用的 CUDA/PyTorch、vLLM 或 LLaMA-Factory 官方镜像。</div>}<div className="model-modal-actions"><button type="button" className="secondary-btn" onClick={() => setImagePickerOpen(false)}>取消</button><button type="button" className="primary-btn" disabled={!pendingYouyunImageId} onClick={() => { const nextImage = youyunImages.find((item) => item.id === pendingYouyunImageId); const nextMode = nextImage?.instanceMode ?? (nextImage?.container === false ? "uhost" : "pod"); setSelectedYouyunImageId(pendingYouyunImageId); setSelectedYouyunMode(nextMode); if (nextImage?.container !== true) { setSelectedPublicModelPath(""); setSelectedModel(""); setModelQuery(""); setModelOptions([]); } setImagePickerOpen(false); setYouyunSpecs(null); setYouyunResource(null); void refreshYouyunSpecs(selectedYouyunGpuType, selectedYouyunRegion, selectedYouyunZone, pendingYouyunImageId, nextMode); }}>使用此镜像</button></div></div></div>}
        {showCreateLeaveConfirm && <div className="training-confirm-backdrop" role="presentation" onClick={() => setShowCreateLeaveConfirm(false)}>
          <div className="training-confirm-modal" role="dialog" aria-modal="true" aria-labelledby="training-leave-title" onClick={(event) => event.stopPropagation()}>
            <div className="training-confirm-icon">!</div>
            <div>
              <h2 id="training-leave-title">确认返回微调任务？</h2>
              <p>当前填写内容尚未提交，返回后将不会保存。</p>
            </div>
            <div className="training-confirm-actions">
              <button type="button" className="secondary-btn" onClick={() => setShowCreateLeaveConfirm(false)}>继续填写</button>
              <button type="button" className="primary-btn" onClick={leaveCreate}>确认返回</button>
            </div>
          </div>
        </div>}
      </section>
    );
  }

  if (view === "detail") {
    if (!detailTask && detailLoading) {
      return <section className="content tuning-content"><div className="empty">正在加载任务详情...</div></section>;
    }
    if (detailError && !detailTask) {
      return <section className="content tuning-content"><div className="form-error">{detailError}</div><button className="secondary-btn" onClick={closeDetail}>返回微调任务</button></section>;
    }
    if (selectedTask && detailTask?.id === selectedId) {
      return <TrainingDetail task={selectedTask} logConnection={detailLogConnection} onBack={closeDetail} onCompare={() => setView("compare")} onStop={() => void handleAction(selectedTask, "停止")} onRetry={() => void handleAction(selectedTask, "重试")} onRelease={() => void handleAction(selectedTask, "结束任务并释放资源")} onDestroy={() => void handleAction(selectedTask, "销毁实例")} onSync={() => void retryArtifactSync(selectedTask.id)} />;
    }
  }

  return (
    <section className="content tuning-content">
      <div className="title-row"><div><h1>微调任务</h1><div className="subtitle">管理 LLaMA-Factory 文本生成 SFT / DPO LoRA 与 QLoRA 训练任务</div></div><button className="primary-btn" onClick={() => setView("create")}>新建微调任务</button></div>
      <div className={`backend-status backend-status-${backendState}`}><span className="status-dot" />{backendState === "online" ? "训练服务已连接 · LLaMA-Factory Worker" : backendState === "checking" ? "正在连接训练服务" : "训练服务未连接，请运行 npm run backend"}</div>
      <div className="summary training-metrics"><div className="metric"><div className="metric-value">{tasks.length}</div><div className="metric-label">全部任务</div></div><div className="metric"><div className="metric-value blue">{tasks.filter((item) => item.state === "running" || item.state === "queued").length}</div><div className="metric-label">运行与排队</div></div><div className="metric"><div className="metric-value green">{tasks.filter((item) => item.state === "completed").length}</div><div className="metric-label">已完成</div></div><div className="metric"><div className="metric-value red">{tasks.filter((item) => item.state === "failed").length}</div><div className="metric-label">失败</div></div></div>
      <div className="filters"><input className="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索任务名称或任务 ID" aria-label="搜索任务名称或任务 ID" /><select className="filter-select" value={stateFilter} onChange={(event) => setStateFilter(event.target.value)}><option value="all">全部状态</option><option value="queued">排队中</option><option value="running">运行中</option><option value="completed">已完成</option><option value="failed">失败</option><option value="stopped">已停止</option></select><div className="filter-spacer" /><button type="button" className="secondary-btn refresh" title="重置筛选" aria-label="重置筛选" onClick={() => { setQuery(""); setStateFilter("all"); }}><RefreshCw size={15} aria-hidden="true" /></button></div>
      <div className="table-wrap training-table"><div className="table-scroll"><table><thead><tr><th>任务名称</th><th>基础模型</th><th>数据集</th><th>训练方式</th><th>状态 / 进度</th><th>创建时间</th><th>操作</th></tr></thead><tbody>{filteredTasks.map((task) => <tr key={task.id}><td><button className="dataset-link" onClick={() => openDetail(task)}>{task.name}</button><div className="row-sub">{task.id}</div></td><td>{task.model}</td><td>{task.dataset}</td><td>{task.method}</td><td><span className={`status ${taskStateMeta[task.state].className}`}><span className="status-dot" />{taskStateMeta[task.state].label}</span>{task.state === "running" && <div className="table-progress"><span style={{ width: `${task.progress}%` }} /></div>}<div className="row-sub">{task.state === "running" ? `${task.progress}% · ${task.currentStep}` : deploymentHint(task)}</div></td><td>{task.createdAt}<div className="row-sub">{task.owner}</div></td><td><div className="actions">{taskActions(task.state).map((action) => <button key={action} disabled={!localRuntimeEnabled && task.provider !== "youyun" && ["重试", "克隆"].includes(action)} title={!localRuntimeEnabled && task.provider !== "youyun" ? LOCAL_RUNTIME_DISABLED : undefined} className={`link-btn ${action === "删除" ? "danger" : ""}`} onClick={() => void handleAction(task, action)}>{action}</button>)}</div></td></tr>)}</tbody></table>{!filteredTasks.length && <div className="empty">{backendState === "online" ? "暂无真实微调任务" : "连接训练服务后显示任务"}</div>}</div><Pagination page={taskPage} totalPages={taskPageInfo.totalPages} total={taskPageInfo.total} unit="条任务" onPageChange={setTaskPage} /></div>
    </section>
  );
}

function ParameterInput({ label, value, onChange, inputType = "integer" }: { label: string; value: string | number; onChange: (value: string) => void; inputType?: "text" | "integer" | "decimal" }) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => {
    const parsedDraft = Number(draft);
    if (typeof value === "number" && Number.isFinite(parsedDraft) && parsedDraft === value) return;
    if (String(value) !== draft) setDraft(String(value));
  }, [value]);

  function handleChange(next: string) {
    setDraft(next);
    if (inputType === "text") {
      onChange(next);
      return;
    }
    if (!next || next === "-" || next === "." || next === "-." || next.endsWith(".")) return;
    const parsed = Number(next);
    if (Number.isFinite(parsed)) onChange(next);
  }

  function handleBlur() {
    if (inputType === "text") return;
    const parsed = Number(draft);
    if (Number.isFinite(parsed)) {
      onChange(String(parsed));
      setDraft(String(parsed));
    } else {
      setDraft(String(value));
    }
  }

  return <label className="parameter-field"><span>{label}</span><input type={inputType === "text" ? "text" : "number"} inputMode={inputType === "decimal" ? "decimal" : inputType === "integer" ? "numeric" : undefined} step={inputType === "decimal" ? "any" : inputType === "integer" ? 1 : undefined} value={draft} onChange={(event) => handleChange(event.target.value)} onBlur={handleBlur} /></label>;
}

function ParameterSelect({ label, value, options, onChange }: { label: string; value: string; options: Array<[string, string]>; onChange: (value: string) => void }) {
  return <label className="parameter-field"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>{options.map(([optionValue, optionLabel]) => <option value={optionValue} key={optionValue}>{optionLabel}</option>)}</select></label>;
}

function ToggleSetting({ label, description, checked, onChange }: { label: string; description: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return <label className="toggle-row setting-toggle"><span><strong>{label}</strong><small>{description}</small></span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><i /></label>;
}

function ChatParameterPanel({ parameters, onChange }: { parameters: ChatParameters; onChange: (parameters: ChatParameters) => void }) {
  return <aside className="panel chat-settings"><div className="chat-settings-head"><div><h2>生成设置</h2><p>基础模型与微调模型使用相同参数</p></div><span className="thinking-off">思考关闭</span></div><div className="chat-parameter-grid">
    <ParameterInput label="最大生成 Token" value={parameters.maxNewTokens} onChange={(value) => onChange({ ...parameters, maxNewTokens: Number(value) })} />
    <ParameterInput inputType="decimal" label="Temperature" value={parameters.temperature} onChange={(value) => onChange({ ...parameters, temperature: Number(value) })} />
    <ParameterInput inputType="decimal" label="Top P" value={parameters.topP} onChange={(value) => onChange({ ...parameters, topP: Number(value) })} />
    <ParameterInput label="随机种子" value={parameters.seed} onChange={(value) => onChange({ ...parameters, seed: Number(value) })} />
  </div></aside>;
}

function ChatTranscript({ messages, assistantLabel }: { messages: ChatMessage[]; assistantLabel: string }) {
  const visibleMessages = messages.filter((message) => message.role !== "system");
  if (!visibleMessages.length) return <div className="chat-empty">开始一轮对话，结果会随训练任务保存。</div>;
  return <div className="chat-thread">{visibleMessages.map((message, index) => <div className={`chat-message chat-message-${message.role}`} key={`${message.role}-${index}`}><div className="chat-speaker">{message.role === "user" ? "你" : assistantLabel}</div><div className="chat-bubble"><p>{message.content}</p>{message.role === "assistant" && (message.elapsedSeconds !== undefined || message.outputTokens !== undefined) && <small>{message.elapsedSeconds?.toFixed(2)} 秒 · {message.outputTokens ?? 0} tokens</small>}</div></div>)}</div>;
}

function BaselineChatWorkspace({ model, tuningMode, datasetId, messages, parameters, onMessagesChange, onParametersChange, onBack }: { model: string; tuningMode: TuningMode; datasetId: string; messages: ChatMessage[]; parameters: ChatParameters; onMessagesChange: (messages: ChatMessage[]) => void; onParametersChange: (parameters: ChatParameters) => void; onBack: () => void }) {
  const { localRuntimeEnabled } = useCapabilities();
  const [split, setSplit] = useState<"test" | "challenge">("test");
  const [prompts, setPrompts] = useState<DatasetPrompt[]>([]);
  const [selectedPromptId, setSelectedPromptId] = useState("");
  const [sampleSystem, setSampleSystem] = useState("");
  const [sampleReference, setSampleReference] = useState("");
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    if (!datasetId) return;
    void apiRequest<DatasetPrompt[]>(`/datasets/${datasetId}/prompts?split=${split}&limit=20`).then((items) => {
      if (!cancelled) {
        setPrompts(items);
        setSelectedPromptId("");
        setSampleReference("");
      }
    }).catch((reason) => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : "读取评测问题失败。");
    });
    return () => { cancelled = true; };
  }, [datasetId, split]);

  const selectPrompt = (promptId: string) => {
    setSelectedPromptId(promptId);
    const prompt = prompts.find((item) => item.id === promptId);
    if (!prompt) return;
    setInput(prompt.prompt);
    setSampleSystem(prompt.system);
    setSampleReference(prompt.reference);
    setError("");
  };

  const send = async () => {
    const content = input.trim();
    if (!content || sending) return;
    const hasSystem = messages.some((message) => message.role === "system");
    const history = !messages.length && sampleSystem && !hasSystem ? [{ role: "system" as const, content: sampleSystem }] : messages;
    const requestMessages: ChatMessage[] = [...history, { role: "user", content }];
    setSending(true);
    setError("");
    try {
      const response = await apiRequest<ChatResponse>("/chat", { method: "POST", body: JSON.stringify({ mode: "base", model, tuningMode, modelSource: "modelscope", messages: requestMessages, parameters }) });
      onMessagesChange([...requestMessages, { role: "assistant", content: response.text, elapsedSeconds: response.elapsedSeconds, outputTokens: response.outputTokens }]);
      setInput("");
      setSelectedPromptId("");
      setSampleReference("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "基模生成失败。");
    } finally {
      setSending(false);
    }
  };

  return <section className="content tuning-content chat-workspace">
    <div className="detail-title-row"><div><BackButton label="返回任务配置" onClick={onBack} /><h1>训练前基模测试</h1><div className="subtitle">{model} · 多轮对话记录将随训练任务保存</div></div><div className="form-actions"><button className="secondary-btn" disabled={sending || !messages.length} onClick={() => { onMessagesChange([]); setError(""); }}>清空对话</button><button className="primary-btn" disabled={sending} onClick={onBack}>保存并返回</button></div></div>
    <div className="chat-layout"><div className="panel chat-main"><div className="sample-toolbar"><select className="filter-select" value={split} onChange={(event) => setSplit(event.target.value as "test" | "challenge")}><option value="test">测试集</option><option value="challenge">挑战集</option></select><select className="sample-select" value={selectedPromptId} onChange={(event) => selectPrompt(event.target.value)}><option value="">从数据集中选择问题</option>{prompts.map((prompt) => <option value={prompt.id} key={prompt.id}>{prompt.prompt}</option>)}</select></div>
      <ChatTranscript messages={messages} assistantLabel="基础模型" />
      {sampleReference && <div className="sample-reference"><strong>数据集参考答案</strong><span>{sampleReference}</span></div>}
      {error && <div className="form-error">{error}</div>}
      <div className="chat-composer"><textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="输入问题，继续当前多轮对话" rows={3} /><button className="primary-btn" title={!localRuntimeEnabled ? LOCAL_RUNTIME_DISABLED : undefined} disabled={!localRuntimeEnabled || sending || !input.trim()} onClick={() => void send()}>{sending ? "生成中" : "发送"}</button></div>
    </div><ChatParameterPanel parameters={parameters} onChange={onParametersChange} /></div>
  </section>;
}

type BaselineTurn = { prompt: string; response: ChatMessage };
type ComparisonTurn = { prompt: string; reference?: string; base?: ChatResponse; tuned?: ChatResponse };
type ComparisonSide = "base" | "adapter" | null;

function baselineTurns(messages: ChatMessage[]): BaselineTurn[] {
  const turns: BaselineTurn[] = [];
  for (let index = 0; index < messages.length; index += 1) {
    if (messages[index].role !== "user") continue;
    const response = messages[index + 1];
    if (response?.role === "assistant") turns.push({ prompt: messages[index].content, response });
  }
  return turns;
}

function ComparisonWorkspace({ task, onBack }: { task: TrainingTask; onBack: () => void }) {
  const { localRuntimeEnabled } = useCapabilities();
  const localDisabled = !localRuntimeEnabled && task.provider !== "youyun";
  const sourceMessages = task.baselineMessages ?? [];
  const [turns, setTurns] = useState<ComparisonTurn[]>(() => baselineTurns(sourceMessages).map((turn) => ({
    prompt: turn.prompt,
    base: {
      text: turn.response.content,
      elapsedSeconds: turn.response.elapsedSeconds ?? 0,
      outputTokens: turn.response.outputTokens ?? 0,
      thinking: false
    }
  })));
  const [split, setSplit] = useState<"train" | "validation" | "test" | "challenge">("train");
  const [prompts, setPrompts] = useState<DatasetPrompt[]>([]);
  const [selectedPromptId, setSelectedPromptId] = useState("");
  const [sampleSystem, setSampleSystem] = useState("");
  const [sampleReference, setSampleReference] = useState("");
  const [sessionSystem, setSessionSystem] = useState("");
  const [input, setInput] = useState("");
  const [generating, setGenerating] = useState(false);
  const [stage, setStage] = useState("");
  const [stageProgress, setStageProgress] = useState(0);
  const [waitingSeconds, setWaitingSeconds] = useState(0);
  const [activeComparisonSide, setActiveComparisonSide] = useState<ComparisonSide>(null);
  const [error, setError] = useState("");
  const progressRef = useRef<HTMLDivElement | null>(null);
  const [adapterVariant, setAdapterVariant] = useState<"best" | "final">(
    task.selectedAdapterVariant ?? (task.bestAdapterPath ? "best" : "final")
  );
  const parameters = task.chatParameters ?? defaultChatParameters;
  const publicModelPath = task.youyunResource?.publicModelPath?.trim() || "";
  const comparisonModelSource = (publicModelPath || task.modelSource === "public") ? "public" : "modelscope";
  const remoteModelLabel = comparisonModelSource === "public" ? "公共基模" : "基模";
  const savedSystem = sourceMessages.find((message) => message.role === "system")?.content ?? "";
  const activeSystem = savedSystem || sessionSystem || (!turns.length ? sampleSystem : "");
  const missingTuned = turns.filter((turn) => turn.base && !turn.tuned).length;
  const adapterLabel = adapterVariant === "best"
    ? (task.selectionMetric === "train_loss" ? "训练 Loss 最佳 Adapter" : "最佳 Adapter")
    : "最终 Adapter";
  const adapterStep = adapterVariant === "best" ? task.bestStep : task.finalStep;
  const adapterLoss = adapterVariant === "best" ? task.bestEvalLoss : task.finalEvalLoss;
  const hasBothAdapters = Boolean(
    task.bestAdapterPath &&
      task.finalAdapterPath &&
      task.bestAdapterPath !== task.finalAdapterPath,
  );
  const youyunInstance = task.artifacts?.find((artifact) => artifact.type === "youyun-instance");
  const isYouyunTask = task.provider === "youyun" || Boolean(youyunInstance);
  const compareUnavailable = isYouyunTask && Boolean(
    !youyunInstance?.instanceId || youyunInstance.state === "released"
  );

  const chooseAdapter = (variant: "best" | "final") => {
    if (variant === adapterVariant) return;
    setAdapterVariant(variant);
    setTurns((items) => items.map((turn) => ({ ...turn, tuned: undefined })));
    setError("");
  };

  useEffect(() => {
    let cancelled = false;
    if (!task.datasetId) return;
    const validationSplit = task.validationSplit ?? 10;
    const seed = task.parameters.seed ?? 42;
    void apiRequest<DatasetPrompt[]>(`/datasets/${task.datasetId}/prompts?split=${split}&limit=50&validation_split=${validationSplit}&seed=${seed}`).then((items) => {
      if (cancelled) return;
      setPrompts(items);
      setSelectedPromptId("");
      setSampleSystem("");
      setSampleReference("");
    }).catch((reason) => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : "读取评测问题失败。");
    });
    return () => { cancelled = true; };
  }, [task.datasetId, split]);

  useEffect(() => {
    if (!generating) {
      setWaitingSeconds(0);
      return;
    }
    const startedAt = Date.now();
    const timer = window.setInterval(() => setWaitingSeconds(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [generating, stage]);

  useEffect(() => {
    // 生成开始或阶段切换时，把对话下方的进度条滚进视野，长对话下也始终可见。
    if (generating) {
      progressRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [generating, stage]);

  const selectPrompt = (promptId: string) => {
    setSelectedPromptId(promptId);
    const prompt = prompts.find((item) => item.id === promptId);
    if (!prompt) {
      setSampleSystem("");
      setSampleReference("");
      return;
    }
    setInput(prompt.prompt);
    setSampleSystem(prompt.system);
    setSampleReference(prompt.reference);
    setError("");
  };

  const historyFor = (items: ComparisonTurn[], side: "base" | "tuned") => {
    const history: ChatMessage[] = activeSystem ? [{ role: "system", content: activeSystem }] : [];
    items.forEach((turn) => {
      const response = side === "base" ? turn.base : turn.tuned;
      if (!response) return;
      history.push({ role: "user", content: turn.prompt });
      history.push({ role: "assistant", content: response.text });
    });
    return history;
  };

  const chatWithWarmupRetry = async (
    body: Record<string, unknown>,
    warmupStageText: string,
  ): Promise<ChatResponse> => {
    // 预热期间后端会快速返回「正在预热」：自动每 5 秒重试，就绪后无缝继续，
    // 演示时点一次发送即可，无需手动二次点击。
    const startedAt = Date.now();
    let warmupNotified = false;
    for (;;) {
      try {
        return await apiRequest<ChatResponse>("/chat", { method: "POST", body: JSON.stringify(body) });
      } catch (reason) {
        const message = reason instanceof Error ? reason.message : String(reason);
        const waitedMs = Date.now() - startedAt;
        if (!message.includes("预热") || waitedMs > 240000) throw reason;
        if (!warmupNotified) {
          warmupNotified = true;
          setStage(warmupStageText);
          setStageProgress(10);
        }
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }
    }
  };

  const generateSavedComparison = async () => {
    setGenerating(true);
    setError("");
    let working = [...turns];
    try {
      const total = working.length;
      for (let index = 0; index < working.length; index += 1) {
        if (working[index].tuned) continue;
        setActiveComparisonSide("adapter");
        setStage(`${isYouyunTask ? "正在优云 GPU 生成" : "正在加载并生成"}第 ${index + 1} / ${total} 轮${adapterLabel}回答`);
        setStageProgress(50 + Math.floor((index / Math.max(total, 1)) * 50));
        const messages = [...historyFor(working.slice(0, index), "tuned"), { role: "user" as const, content: working[index].prompt }];
        const tuned = await chatWithWarmupRetry({ mode: "adapter", adapterVariant, taskId: task.id, model: task.model, modelSource: comparisonModelSource, messages, parameters }, `优云 GPU 正在预热（连接实例并加载${remoteModelLabel}与 Adapter），就绪后自动继续生成回答`);
        working = working.map((turn, turnIndex) => turnIndex === index ? { ...turn, tuned } : turn);
        setTurns(working);
        setStageProgress(50 + Math.floor(((index + 1) / Math.max(total, 1)) * 50));
      }
      setStage("训练后回答生成完成");
      setStageProgress(100);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "微调模型生成失败。");
    } finally {
      setGenerating(false);
      setActiveComparisonSide(null);
      setStage("");
      setStageProgress(0);
    }
  };

  const sendNewComparison = async () => {
    const prompt = input.trim();
    if (!prompt || generating || missingTuned) return;
    setGenerating(true);
    setError("");
    setInput("");
    const index = turns.length;
    let working: ComparisonTurn[] = [...turns, { prompt, reference: sampleReference || undefined }];
    if (!turns.length && sampleSystem && !savedSystem) setSessionSystem(sampleSystem);
    setTurns(working);
    setSelectedPromptId("");
    setSampleSystem("");
    setSampleReference("");
    try {
      setActiveComparisonSide("base");
      setStage(isYouyunTask
        ? `正在优云 GPU 生成基础模型回答（必要时连接实例并加载${remoteModelLabel}）`
        : `正在加载基模与${adapterLabel}，并生成基础模型回答`);
      setStageProgress(25);
      const baseMessages = [...historyFor(turns, "base"), { role: "user" as const, content: prompt }];
      const base = await chatWithWarmupRetry({ mode: "base", taskId: task.id, adapterVariant, model: task.model, modelSource: comparisonModelSource, messages: baseMessages, parameters }, `优云 GPU 正在预热（连接实例并加载${remoteModelLabel}与 Adapter），就绪后自动生成基础模型回答`);
      working = working.map((turn, turnIndex) => turnIndex === index ? { ...turn, base } : turn);
      setTurns(working);
      setStageProgress(50);
      setActiveComparisonSide("adapter");
      setStage(`基础模型回答完成，正在生成${adapterLabel}回答`);
      setStageProgress(75);
      const tunedMessages = [...historyFor(turns, "tuned"), { role: "user" as const, content: prompt }];
      const tuned = await chatWithWarmupRetry({ mode: "adapter", adapterVariant, taskId: task.id, model: task.model, modelSource: comparisonModelSource, messages: tunedMessages, parameters }, `优云 GPU 正在预热（连接实例并加载${remoteModelLabel}与 Adapter），就绪后自动生成 Adapter 回答`);
      working = working.map((turn, turnIndex) => turnIndex === index ? { ...turn, tuned } : turn);
      setTurns(working);
      setStage("训练前后回答生成完成");
      setStageProgress(100);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "训练前后对比生成失败。");
    } finally {
      setGenerating(false);
      setActiveComparisonSide(null);
      setStage("");
      setStageProgress(0);
    }
  };

  const stageWithTime = generating ? `${stage} · ${waitingSeconds} 秒` : "";

  return <section className="content tuning-content compare-workspace"><div className="detail-title-row"><div><BackButton label="返回任务详情" onClick={onBack} /><h1>训练前后效果对比</h1><div className="subtitle">{task.name} · {task.model} · 当前加载{adapterLabel}{typeof adapterStep === "number" ? `（Step ${adapterStep}）` : ""} · 思考模式关闭</div></div><div className="compare-adapter-actions">{hasBothAdapters && <div className="segmented adapter-segmented" aria-label="选择对比 Adapter"><button type="button" className={adapterVariant === "best" ? "selected" : ""} title={localDisabled ? LOCAL_RUNTIME_DISABLED : undefined} disabled={localDisabled || generating} onClick={() => chooseAdapter("best")}>最佳</button><button type="button" className={adapterVariant === "final" ? "selected" : ""} title={localDisabled ? LOCAL_RUNTIME_DISABLED : undefined} disabled={localDisabled || generating} onClick={() => chooseAdapter("final")}>最终</button></div>}{missingTuned > 0 && <button className="primary-btn" title={localDisabled ? LOCAL_RUNTIME_DISABLED : undefined} disabled={localDisabled || generating || compareUnavailable} onClick={() => void generateSavedComparison()}>{generating ? stageWithTime : `生成 ${missingTuned} 轮${adapterLabel}回答`}</button>}</div></div>
    <div className="adapter-selection-note"><strong>当前对比 Adapter</strong><span>{adapterLabel}{typeof adapterStep === "number" ? ` · Step ${adapterStep}` : ""}{typeof adapterLoss === "number" ? ` · 验证 Loss ${adapterLoss.toFixed(4)}` : ""}</span></div>
    {compareUnavailable && <div className="form-error" role="alert">优云实例已释放，效果对比不可用；重新运行任务可获得新的对比实例。</div>}
    {!turns.length && <div className="panel compare-empty"><h2>开始新的对比对话</h2><p>从评测集中选择或自行输入问题，系统会依次生成基础模型和 LoRA 模型回答。</p></div>}
    <div className="comparison-list">{turns.map((turn, index) => <section className="panel comparison-turn" key={`${turn.prompt}-${index}`}><div className="comparison-prompt"><span>第 {index + 1} 轮问题</span><strong>{turn.prompt}</strong>{turn.reference && <div className="sample-reference"><strong>预期答案</strong><span>{turn.reference}</span></div>}</div><div className="comparison-columns"><div className="comparison-answer base-answer"><div className="comparison-label"><span>训练前</span><strong>基础模型</strong></div>{turn.base ? <><p>{turn.base.text}</p><small>{turn.base.elapsedSeconds.toFixed(2)} 秒 · {turn.base.outputTokens} tokens</small></> : <div className="comparison-waiting">{generating && activeComparisonSide === "base" ? stageWithTime : generating ? "等待基础模型生成" : "等待生成"}</div>}</div><div className="comparison-answer tuned-answer"><div className="comparison-label"><span>训练后</span><strong>{adapterLabel}</strong></div>{turn.tuned ? <><p>{turn.tuned.text}</p><small>{turn.tuned.elapsedSeconds.toFixed(2)} 秒 · {turn.tuned.outputTokens} tokens{typeof turn.tuned.adapterStep === "number" ? ` · 已加载 Step ${turn.tuned.adapterStep}` : ""}</small></> : <div className="comparison-waiting">{error ? `${adapterLabel}推理未完成：${error}` : generating && activeComparisonSide === "adapter" ? stageWithTime : generating ? "等待基础模型回答完成" : "等待生成"}</div>}</div></div></section>)}</div>
    {generating && <div ref={progressRef} className="comparison-progress" role="status" aria-live="polite"><div className="comparison-progress-head"><strong>{stage}</strong><span>已等待 {waitingSeconds} 秒</span></div><div className="comparison-progress-track"><span style={{ width: `${stageProgress}%` }} /></div><div className="comparison-progress-meta"><b>{stageProgress}%</b></div></div>}
    {error && <div className="form-error">{error}</div>}
    <section className="panel"><div className="sample-toolbar"><select className="filter-select" value={split} title={localDisabled ? LOCAL_RUNTIME_DISABLED : undefined} disabled={localDisabled || generating} onChange={(event) => setSplit(event.target.value as "train" | "validation" | "test" | "challenge")}><option value="train">训练集</option><option value="validation">验证集</option><option value="test">测试集（后续评测）</option><option value="challenge">挑战集（后续评测）</option></select><select className="sample-select" value={selectedPromptId} title={localDisabled ? LOCAL_RUNTIME_DISABLED : undefined} disabled={localDisabled || generating || !prompts.length} onChange={(event) => selectPrompt(event.target.value)}><option value="">{prompts.length ? "从当前分片选择问题" : "当前数据集未提供该分片"}</option>{prompts.map((prompt) => <option value={prompt.id} key={prompt.id}>{prompt.prompt}</option>)}</select></div>{sampleReference && <div className="sample-reference"><strong>数据集预期答案</strong><span>{sampleReference}</span></div>}</section>
    <section className="panel"><div className="chat-composer"><textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder={missingTuned ? "先生成已保存对话的微调后回答" : compareUnavailable ? "优云实例已释放，无法继续对比" : "输入问题，继续训练前后多轮对比"} rows={3} title={localDisabled ? LOCAL_RUNTIME_DISABLED : undefined} disabled={localDisabled || generating || missingTuned > 0 || compareUnavailable} /><button className="primary-btn" title={localDisabled ? LOCAL_RUNTIME_DISABLED : undefined} disabled={localDisabled || generating || missingTuned > 0 || compareUnavailable || !input.trim()} onClick={() => void sendNewComparison()}>{generating ? stage : "发送并对比"}</button></div></section>
  </section>;
}

function LossChart({ metrics }: { metrics: TrainingMetric[] }) {
  const trainPoints = metrics
    .filter((metric) => typeof metric.trainLoss === "number")
    .map((metric) => ({ step: metric.step, value: metric.trainLoss as number }));
  const evalPoints = metrics
    .filter((metric) => typeof metric.evalLoss === "number")
    .map((metric) => ({ step: metric.step, value: metric.evalLoss as number }));
  const allPoints = [...trainPoints, ...evalPoints];

  if (!allPoints.length) {
    return <section className="panel chart-card"><div className="panel-head"><div><h2>Loss 曲线</h2><p>训练日志产生指标后自动绘制</p></div></div><div className="loss-empty">暂无 Loss 记录</div></section>;
  }

  const width = 720;
  const height = 220;
  const plot = { left: 58, right: 700, top: 18, bottom: 180 };
  const steps = allPoints.map((point) => point.step);
  const values = allPoints.map((point) => point.value);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const padding = rawMin === rawMax ? Math.max(Math.abs(rawMin) * 0.08, 0.1) : (rawMax - rawMin) * 0.12;
  const minValue = rawMin - padding;
  const maxValue = rawMax + padding;
  const x = (step: number) => minStep === maxStep
    ? (plot.left + plot.right) / 2
    : plot.left + ((step - minStep) / (maxStep - minStep)) * (plot.right - plot.left);
  const y = (value: number) => plot.bottom - ((value - minValue) / (maxValue - minValue)) * (plot.bottom - plot.top);
  const polyline = (points: Array<{ step: number; value: number }>) => points.map((point) => `${x(point.step)},${y(point.value)}`).join(" ");
  const yTicks = Array.from({ length: 5 }, (_, index) => maxValue - ((maxValue - minValue) * index) / 4);

  return <section className="panel chart-card">
    <div className="panel-head"><div><h2>Loss 曲线</h2><p>来自 LLaMA-Factory Trainer 的真实日志指标</p></div><div className="chart-legends">{trainPoints.length > 0 && <span className="chart-legend"><i />训练 Loss</span>}{evalPoints.length > 0 && <span className="chart-legend eval"><i />验证 Loss</span>}</div></div>
    <div className="loss-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="训练 Loss 曲线">
        {yTicks.map((tick) => {
          const tickY = y(tick);
          return <g key={tick}><line x1={plot.left} x2={plot.right} y1={tickY} y2={tickY} /><text x={plot.left - 10} y={tickY + 4} textAnchor="end">{tick.toFixed(4)}</text></g>;
        })}
        {trainPoints.length > 1 && <polyline className="train-series" points={polyline(trainPoints)} />}
        {evalPoints.length > 1 && <polyline className="eval-series" points={polyline(evalPoints)} />}
        {trainPoints.map((point) => <circle className="train-point" key={`train-${point.step}`} cx={x(point.step)} cy={y(point.value)} r="4"><title>Step {point.step} · 训练 Loss {point.value.toFixed(4)}</title></circle>)}
        {evalPoints.map((point) => <circle className="eval-point" key={`eval-${point.step}`} cx={x(point.step)} cy={y(point.value)} r="4"><title>Step {point.step} · 验证 Loss {point.value.toFixed(4)}</title></circle>)}
        <text className="axis-label" x={x(minStep)} y={207} textAnchor={minStep === maxStep ? "middle" : "start"}>Step {minStep}</text>
        {minStep !== maxStep && <text className="axis-label" x={x(maxStep)} y={207} textAnchor="end">Step {maxStep}</text>}
      </svg>
    </div>
    {allPoints.length === 1 && <div className="chart-note">当前仅有 1 个 Loss 记录点，后续训练日志会自动形成曲线。</div>}
  </section>;
}

function TrainingDetail({ task: incomingTask, logConnection, onBack, onCompare, onStop, onRetry, onRelease, onDestroy, onSync }: { task: TrainingTask; logConnection: LogConnectionState; onBack: () => void; onCompare: () => void; onStop: () => void; onRetry: () => void; onRelease: () => void; onDestroy: () => void; onSync: () => void }) {
  const { localRuntimeEnabled } = useCapabilities();
  const localDisabled = !localRuntimeEnabled && incomingTask.provider !== "youyun";
  const youyunInstance = incomingTask.artifacts?.find((artifact) => artifact.type === "youyun-instance");
  const task: TrainingTask = {
    ...incomingTask,
    artifacts: incomingTask.artifacts?.filter((artifact) => artifact.type !== "youyun-instance")
  };
  const [openingPath, setOpeningPath] = useState("");
  const [preview, setPreview] = useState<ArtifactPreview | null>(null);
  const [previewFile, setPreviewFile] = useState<ArtifactPreviewFileResult | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const completed = task.state === "completed";
  const finished = completed || task.state === "stopped";
  const hasBaseline = task.baselineMessages?.some((message) => message.role === "assistant") ?? false;
  const hasAdapter = task.artifacts?.some((artifact) => artifact.type === "adapter") ?? false;
  const hasMergedModel = task.artifacts?.some((artifact) => artifact.type === "merged") ?? false;
  const instanceRetained = Boolean(youyunInstance?.instanceId && youyunInstance.state !== "released");
  const isYouyunTask = incomingTask.provider === "youyun" || Boolean(youyunInstance);
  const nodeCopy = trainingNodeCopy(task);
  const metricLabel = task.selectionMetric === "train_loss" ? "训练 Loss" : "验证 Loss";
  const evaluations = [
    { label: "最佳 Step", value: task.bestStep?.toString() ?? "--" },
    { label: `最佳${metricLabel}`, value: task.bestLoss?.toFixed(4) ?? task.bestEvalLoss?.toFixed(4) ?? "--" },
    { label: `最终${metricLabel}`, value: task.finalLoss?.toFixed(4) ?? task.finalEvalLoss?.toFixed(4) ?? "--" },
    { label: "最终 Step", value: task.finalStep?.toString() ?? "--" }
  ];
  const openArtifact = async (path: string) => {
    setOpeningPath(path);
    setPreviewLoading(true);
    setPreviewError("");
    setPreviewFile(null);
    try {
      const result = await apiRequest<ArtifactPreview | ArtifactPreviewFileResult>("/artifacts/preview", { method: "POST", body: JSON.stringify({ path }) });
      if ("files" in result) {
        setPreview(result);
      } else {
        setPreview({ path, name: path.split(/[\\/]/).pop() || "训练产物", files: [] });
        setPreviewFile(result);
      }
    } catch (error) {
      setPreviewError(error instanceof Error ? error.message : "无法读取产物目录");
      setPreview({ path, name: "训练产物", files: [] });
    } finally {
      setOpeningPath("");
      setPreviewLoading(false);
    }
  };
  const openArtifactFile = async (file: ArtifactPreviewFile) => {
    if (!preview) return;
    setPreviewLoading(true);
    setPreviewError("");
    setPreviewFile(null);
    try {
      const result = await apiRequest<ArtifactPreviewFileResult>("/artifacts/preview", { method: "POST", body: JSON.stringify({ path: preview.path, relativePath: file.path }) });
      setPreviewFile(result);
    } catch (error) {
      setPreviewError(error instanceof Error ? error.message : "无法读取产物文件");
    } finally {
      setPreviewLoading(false);
    }
  };
  const scrollToTop = () => {
    document.querySelector<HTMLElement>(".main")?.scrollTo({ top: 0, behavior: "smooth" });
  };
  const bestAdapter = task.artifacts?.find((artifact) => artifact.type === "adapter" && artifact.variant === "best");
  const finalAdapter = task.artifacts?.find((artifact) => artifact.type === "adapter" && artifact.variant === "final");
  const syncingAdapters = task.artifacts?.filter((artifact) => artifact.type === "adapter" && artifact.syncState === "syncing").length ?? 0;
  const failedAdapters = task.artifacts?.filter((artifact) => artifact.type === "adapter" && (artifact.syncState === "failed" || Boolean(artifact.syncError))).length ?? 0;
  const remoteReadyAdapters = task.artifacts?.filter((artifact) => artifact.type === "adapter" && artifact.syncState === "remote_ready").length ?? 0;
  const duplicateAdapter = Boolean(
    bestAdapter &&
      finalAdapter &&
      (bestAdapter.path === finalAdapter.path ||
        (task.selectionMetric === "train_loss" &&
          bestAdapter.checkpointStep != null &&
          bestAdapter.checkpointStep === finalAdapter.checkpointStep)),
  );
  const displayLogs = task.logs?.length
    ? normalizeInferenceLogs({ logs: task.logs })
    : [];
  const logConnectionCopy = finished
    ? "日志已保存"
    : logConnection === "live"
      ? "WebSocket 实时同步"
      : logConnection === "fallback"
        ? "连接中断，已切换为每 2 秒自动刷新"
        : "正在建立实时连接";
  const visibleArtifacts = task.artifacts?.filter((artifact) => !(duplicateAdapter && artifact.type === "adapter" && artifact.variant === "final"));
  const modelArtifacts = visibleArtifacts?.filter((artifact) => artifact.type === "adapter" || artifact.type === "merged");
  const auxiliaryArtifacts = visibleArtifacts?.filter((artifact) => artifact.type === "metrics" || artifact.type === "manifest");
  const renderArtifact = (artifact: NonNullable<TrainingTask["artifacts"]>[number]) => { const isSelected = artifact.path === task.selectedAdapterPath; const duplicateLabel = duplicateAdapter && artifact.type === "adapter" && artifact.variant === "best"; const info = trainingArtifactInfo(artifact, task, duplicateLabel, isSelected); return <div className={`artifact ${isSelected ? "selected-artifact" : ""}`} key={artifact.path}><span className={`artifact-icon ${artifact.type === "merged" ? "full" : ""}`}>{info.icon}</span><div><strong>{info.label}</strong><small>{artifact.name} · {info.description}</small><button className="artifact-path" title="查看产物文件" disabled={openingPath === artifact.path} onClick={() => void openArtifact(artifact.path)}>{artifact.path}</button></div><button className="secondary-btn compact-btn" disabled={openingPath === artifact.path} onClick={() => void openArtifact(artifact.path)}>{openingPath === artifact.path ? "正在读取" : "查看产物"}</button></div>; };
  return <section className="content tuning-content">
    <div className="detail-title-row"><div><BackButton label="返回微调任务列表" onClick={onBack} /><div className="detail-title-line"><h1>{task.name}</h1><span className={`status ${taskStateMeta[task.state].className}`}><span className="status-dot" />{taskStateMeta[task.state].label}</span></div><div className="subtitle">{task.id} · 创建于 {task.createdAt}</div></div><div className="form-actions">{finished && hasAdapter && <button className="primary-btn" onClick={onCompare}>{hasBaseline ? "查看效果对比" : "开始效果对比"}</button>}{finished && instanceRetained && <button className="secondary-btn" onClick={onRelease}>结束任务并释放资源</button>}{finished && instanceRetained && <button className="danger-btn" onClick={onDestroy}>销毁实例</button>}{isActiveTrainingState(task.state) && <button className="danger-btn" onClick={onStop}>停止任务</button>}{(task.state === "failed" || task.state === "stopped") && !instanceRetained && <button className="primary-btn" disabled={localDisabled} title={localDisabled ? LOCAL_RUNTIME_DISABLED : undefined} onClick={onRetry}>重新运行</button>}</div></div>
    <div className="task-overview-grid"><div className="panel progress-card"><div className="panel-head"><div><span className="summary-eyebrow">训练进度</span><h2>{task.currentStep}</h2></div><strong>{task.progress}%</strong></div><div className="large-progress"><span style={{ width: `${task.progress}%` }} /></div><div className="progress-meta"><span>状态由 {nodeCopy.worker} 实时回传</span><span>{completed ? "训练完成" : task.state === "stopped" ? "已保存早停产物" : "自动刷新"}</span></div>{task.stopReason && <div className="row-sub">结束原因：{task.stopReason === "automatic" ? "自动早停" : "用户手动停止"}</div>}{task.error && <div className="form-error">{task.error}</div>}</div>{evaluations.map((metric) => <div className="panel result-metric" key={metric.label}><span>{metric.label}</span><strong>{metric.value}</strong><small>{task.selectionMetric === "train_loss" ? "基于训练 Loss" : task.validationSplit ? "验证检查点指标" : "未启用验证"}</small></div>)}</div>
    {task.youyunResource && <section className="panel youyun-task-resource"><div><strong>远程优云 GPU</strong><span>{task.youyunResource.gpuType} · {task.youyunResource.gpuCount} 卡 · {task.youyunResource.cpuCores} 核 / {task.youyunResource.memoryGb}GB 内存</span></div><div><strong>预计费用</strong><span>¥{task.youyunResource.price.total.toFixed(2)} / 小时 · {task.youyunResource.chargeType === "Postpay" ? "按量计费" : task.youyunResource.chargeType}</span></div></section>}
    <div className="detail-columns"><div className="detail-main-column">
      <LossChart metrics={task.lossHistory ?? []} />
      <section className="panel logs-card"><div className="panel-head"><div><h2>运行日志</h2><p>来自{nodeCopy.node} · {logConnectionCopy}</p></div></div><pre>{displayLogs.length ? displayLogs.join("\n") : "等待 Worker 输出日志..."}</pre></section></div>
       <aside className="detail-side-column"><div className="detail-side-actions"><BackButton label="返回任务列表" onClick={onBack} /><button type="button" className="secondary-btn compact-btn detail-top-btn" onClick={scrollToTop} title="回到当前页面顶部"><ArrowUp aria-hidden="true" size={15} />回到顶部</button></div><section className="panel snapshot-card"><div className="snapshot-card-head"><div><span className="summary-eyebrow">CONFIGURATION</span><h2>配置快照</h2><p>提交时保存的任务配置</p></div><span className="snapshot-card-badge">已锁定</span></div><div className="snapshot-groups"><div className="snapshot-group"><h3>基础配置</h3><dl className="snapshot-grid"><div><dt>基础模型</dt><dd className="snapshot-value-code" title={task.model}>{task.model}</dd></div><div><dt>数据集</dt><dd className="snapshot-value-code" title={task.dataset}>{task.dataset}</dd></div><div><dt>训练框架</dt><dd><span className="snapshot-tag">LLaMA-Factory</span></dd></div><div><dt>训练方式</dt><dd><span className="snapshot-tag snapshot-tag-brand">{task.method}</span></dd></div></dl></div><div className="snapshot-group"><h3>训练超参</h3><dl className="snapshot-grid"><div><dt>Epochs</dt><dd>{task.parameters.epochs}</dd></div><div><dt>学习率</dt><dd className="snapshot-value-code">{task.parameters.learningRate}</dd></div><div><dt>Batch / 累积</dt><dd>{task.parameters.batchSize} / {task.parameters.gradientAccumulation}</dd></div><div><dt>最大序列</dt><dd>{task.parameters.maxSequenceLength}</dd></div><div><dt>LoRA Rank / Alpha</dt><dd>{task.parameters.loraRank} / {task.parameters.loraAlpha}</dd></div><div><dt>作用模块</dt><dd className="snapshot-value-code">{task.parameters.targetModules}</dd></div></dl></div><div className="snapshot-group"><h3>验证与检查点</h3><dl className="snapshot-grid"><div><dt>验证拆分</dt><dd>{task.validationSplit ?? 10}%</dd></div><div><dt>检查点</dt><dd>最多 {task.checkpointLimit ?? 2} 个</dd></div></dl></div></div></section>
      <section className="panel output-card"><h2>训练输出</h2>{finished && isYouyunTask && (syncingAdapters > 0 || failedAdapters > 0 || remoteReadyAdapters > 0) && <div className="artifact-sync-note"><strong>效果对比已可用</strong><span>{syncingAdapters > 0 ? "权重正在后台回传，本地文件暂不可预览。" : failedAdapters > 0 ? "权重同步失败，可重试；远程实例仍可用于效果对比。" : "权重已回传，可从本地镜像预览。"}</span>{failedAdapters > 0 && <button type="button" className="link-btn" onClick={onSync}>重新同步产物</button>}</div>}{finished && task.selectedAdapterPath && <div className="adapter-selection-note compact"><strong>默认用于聊天</strong><span>{task.selectedAdapterVariant === "best" ? (task.selectionMetric === "train_loss" ? "训练 Loss 最佳 Adapter" : "最佳 Adapter") : "最终 Adapter"}{task.selectedAdapterVariant === "best" && typeof task.bestStep === "number" ? ` · Step ${task.bestStep}` : typeof task.finalStep === "number" ? ` · Step ${task.finalStep}` : ""}</span><small>{task.selectionReason}</small></div>}{modelArtifacts?.length ? modelArtifacts.map(renderArtifact) : <div className="artifact"><span className="artifact-icon">A</span><div><strong>LoRA Adapter</strong><small>{task.outputName ?? "训练完成后生成"}</small></div></div>}{Boolean(auxiliaryArtifacts?.length) && <details className="artifact-supporting"><summary>辅助文件（训练指标与产物清单）</summary>{auxiliaryArtifacts?.map(renderArtifact)}</details>}{finished && !hasMergedModel && <div className="artifact-message">本任务未勾选“训练完成后合并模型”，因此不生成合并模型。</div>}<div className="output-path">LLaMA-Factory 工作目录 / {task.id}</div></section></aside></div>
    {preview && <div className="model-modal-layer preview-layer" role="dialog" aria-modal="true" aria-label="训练产物预览"><div className="model-modal preview-modal artifact-preview-modal"><div className="model-modal-head"><div><span className="drawer-kicker">训练产物预览</span><h2>{preview.name}</h2><p>{preview.path}</p></div><button type="button" className="close-btn" onClick={() => { setPreview(null); setPreviewFile(null); }} aria-label="关闭预览"><X size={18} /></button></div>{previewError && <div className="form-error">{previewError}</div>}{previewLoading && !previewFile ? <div className="model-empty">正在读取产物文件...</div> : previewFile ? <div className="preview-content-wrap"><div className="preview-file-head">{preview.files.length > 0 && <button type="button" className="link-btn" onClick={() => setPreviewFile(null)}>返回文件列表</button>}<strong>{previewFile.path}</strong></div>{previewFile.kind === "text" ? <pre className="preview-code">{previewFile.content}</pre> : <div className="preview-binary"><strong>二进制模型文件</strong><p>{previewFile.message ?? "平台不直接展示原始权重内容。"}</p><span>大小：{previewFile.sizeBytes} bytes</span></div>}</div> : <div className="preview-file-list">{preview.files.map((file) => <button type="button" className="preview-file-row" key={file.path} onClick={() => void openArtifactFile(file)}><span><strong><FileText size={13} aria-hidden="true" />{file.path}</strong><small>{file.modifiedAt ?? ""}</small></span><em>{file.sizeBytes} bytes</em></button>)}{!preview.files.length && <div className="model-empty">目录中没有可预览文件</div>}</div>}{(preview.truncated || previewFile?.truncated) && <p className="drawer-help">{previewFile ? "文件内容较大，仅预览前 512 KiB。" : "文件较多，仅展示前 200 个。"}</p>}</div></div>}
  </section>;
}
