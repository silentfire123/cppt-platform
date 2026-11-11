export type DatasetState = "failed" | "ready" | "published";
export type DatasetType = "文本生成" | "文生图";
export type DatasetPurpose = "training" | "evaluation";

export type Dataset = {
  id: string;
  name: string;
  subtitle: string;
  type: DatasetType;
  structure?: "SFT" | "DPO";
  purpose?: DatasetPurpose;
  count: string;
  state: DatasetState;
  validationMessage?: string;
  validationReport?: DatasetValidationReport;
  repairHistory?: DatasetRepairHistoryItem[];
  version?: number;
  warningCount?: number;
  updatedAt: string;
  owner: string;
};

export type DatasetValidationIssue = {
  code: string;
  severity: "error" | "warning";
  line?: number | null;
  path?: string;
  message: string;
  fixable?: boolean;
  suggestion?: string;
};

export type DatasetValidationReport = {
  status?: "failed" | "ready";
  checkedRows?: number;
  validRows?: number;
  invalidRows?: number;
  errors?: DatasetValidationIssue[];
  warnings?: DatasetValidationIssue[];
  autoFixableCount?: number;
  validatorVersion?: string;
  checkedAt?: string;
};

export type DatasetRepairHistoryItem = {
  mode: "auto" | "manual";
  timestamp: string;
  fromVersion: number;
  toVersion: number;
  stats?: Record<string, number>;
};

export type DatasetRecordItem = {
  index: number;
  record: Record<string, unknown>;
};

export type DatasetRecordsResponse = {
  items: DatasetRecordItem[];
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
};

export type TrainingState = "queued" | "provisioning" | "preparing" | "running" | "stopping" | "collecting" | "uploading_artifact" | "completed" | "failed" | "stopped";
export type TuningMode = "lora" | "qlora";
export type TrainingTaskType = "SFT" | "DPO";

export type TrainingArtifact = {
  type: "adapter" | "merged" | "metrics" | "manifest" | "youyun-instance";
  name: string;
  path: string;
  remotePath?: string;
  syncState?: "remote_ready" | "syncing" | "available" | "failed";
  syncError?: string | null;
  variant?: "best" | "final";
  selected?: boolean;
  checkpointStep?: number | null;
  evalLoss?: number | null;
  trainLoss?: number | null;
  loss?: number | null;
  metric?: "eval_loss" | "train_loss";
  sourceAdapterVariant?: "best" | "final";
  provider?: "youyun";
  instanceId?: string;
  gpuType?: string;
  gpuCount?: number;
  zone?: string;
  state?: "running" | "awaiting_comparison" | "debugging" | "releasing" | "released" | "cleanup_failed";
  cleanupError?: string;
};

export type TrainingMetric = {
  step: number;
  epoch?: number;
  trainLoss?: number;
  evalLoss?: number;
  learningRate?: number;
};

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
  elapsedSeconds?: number;
  outputTokens?: number;
};

export type ChatParameters = {
  maxNewTokens: number;
  temperature: number;
  topP: number;
  seed: number;
};

export type ChatResponse = {
  text: string;
  elapsedSeconds: number;
  outputTokens: number;
  thinking: false;
  adapterVariant?: "best" | "final";
  adapterStep?: number | null;
  adapterPath?: string;
};

export type DatasetPrompt = {
  id: string;
  prompt: string;
  system: string;
  reference: string;
};

export type TrainingParameters = {
  epochs: number;
  learningRate: string;
  batchSize: number;
  gradientAccumulation: number;
  validationSteps: number;
  maxSequenceLength: number;
  loraRank: number;
  loraAlpha: number;
  loraDropout: number;
  lrScheduler: "linear" | "cosine" | "constant";
  loraPlusEnabled: boolean;
  loraPlusRatio: number;
  targetModules: "auto" | "attention" | "all-linear";
  optimizer: "adamw_8bit" | "adamw_torch" | "paged_adamw_8bit";
  warmupRatio: number;
  weightDecay: number;
  maxGradNorm: number;
  seed: number;
  loggingSteps: number;
  packing: boolean;
  gradientCheckpointing: boolean;
};

export type TrainingTask = {
  id: string;
  name: string;
  framework?: "llamafactory";
  executionMode?: "local" | "remote";
  provider?: "local" | "youyun";
  modelSource?: "public" | "modelscope" | "local";
  youyunResource?: {
    mode: "auto" | "manual";
    gpuType: string;
    gpuCount: number;
    cpuCores: number;
    memoryGb: number;
    imageId?: string;
    imageName?: string | null;
    imageVersion?: string | null;
    publicModelPath?: string | null;
    region: string;
    zone: string;
    chargeType: string;
    price: { currency: string; instance: number; disks: number; systemDisks: number; image: number; total: number };
  } | null;
  datasetVersion?: string | null;
  state: TrainingState;
  model: string;
  dataset: string;
  method: "SFT / LoRA" | "SFT / QLoRA" | "DPO / LoRA" | "DPO / QLoRA";
  taskType?: TrainingTaskType;
  tuningMode: TuningMode;
  progress: number;
  currentStep: string;
  loss?: number;
  createdAt: string;
  owner: string;
  mergeModel: boolean;
  validationSplit?: number;
  outputName?: string;
  checkpointLimit?: number;
  checkpointStrategy?: "epoch" | "steps";
  checkpointInterval?: number;
  maxSteps?: number;
  earlyStopping?: boolean;
  patience?: number;
  minDelta?: number;
  extraParameters?: Record<string, unknown>;
  parameters: TrainingParameters;
  datasetId?: string;
  error?: string;
  logs?: string[];
  artifacts?: TrainingArtifact[];
  lossHistory?: TrainingMetric[];
  baselineMessages?: ChatMessage[];
  chatParameters?: ChatParameters;
  bestStep?: number | null;
  bestEvalLoss?: number | null;
  finalEvalLoss?: number | null;
  selectionMetric?: "eval_loss" | "train_loss" | null;
  bestLoss?: number | null;
  finalLoss?: number | null;
  stopReason?: "manual" | "automatic" | "manual_timeout" | null;
  finalStep?: number | null;
  selectedAdapterPath?: string | null;
  selectedAdapterVariant?: "best" | "final" | null;
  bestAdapterPath?: string | null;
  finalAdapterPath?: string | null;
  selectionReason?: string | null;
};
