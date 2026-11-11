// Static demo fixtures; values are examples, not production metrics.
import { Dataset, TrainingTask } from "./types";

export const initialDatasets: Dataset[] = [
  { id: "qwenruozhi_v1", name: "平台演示 SFT v1", subtitle: "JSONL · train / val / test / challenge", type: "文本生成", structure: "SFT", count: "4 条", state: "published", validationMessage: "合成演示样例，仅用于格式和流程验证。", updatedAt: "2026-08-15 14:30", owner: "平台管理员" },
  { id: "ds-3", name: "客服偏好 DPO", subtitle: "prompt / chosen / rejected", type: "文本生成", structure: "DPO", count: "3,260 组", state: "failed", validationMessage: "第 128 行 chosen 字段为空。", updatedAt: "2026-08-14 11:06", owner: "李悦" },
  { id: "ds-4", name: "内部知识指令", subtitle: "JSONL · 校验通过，可发布", type: "文本生成", structure: "SFT", count: "8,940 条", state: "ready", updatedAt: "2026-08-13 18:20", owner: "刘士博" },
  { id: "ds-5", name: "产品海报 LoRA", subtitle: "metadata.jsonl · 186 个样本", type: "文生图", count: "186 张", state: "published", updatedAt: "2026-08-12 16:45", owner: "张敏" }
];

const defaultParameters = {
  epochs: 3,
  learningRate: "2e-4",
  batchSize: 1,
  gradientAccumulation: 8,
  validationSteps: 50,
  maxSequenceLength: 1024,
  loraRank: 16,
  loraAlpha: 16,
  loraDropout: 0,
  lrScheduler: "linear" as const,
  loraPlusEnabled: false,
  loraPlusRatio: 16,
  targetModules: "auto" as const,
  optimizer: "adamw_torch" as const,
  warmupRatio: 0.1,
  weightDecay: 0.01,
  maxGradNorm: 1,
  seed: 42,
  loggingSteps: 10,
  packing: false,
  gradientCheckpointing: true
};

export const initialTrainingTasks: TrainingTask[] = [
  {
    id: "ft-240815-03",
    name: "Qwen3 平台演示 SFT v1",
    state: "running",
    model: "Qwen/Qwen3-1.7B",
    dataset: "qwenruozhi_v1",
    method: "SFT / LoRA",
    tuningMode: "lora",
    progress: 64,
    currentStep: "Epoch 2/3 · Step 768/1196",
    loss: 1.184,
    createdAt: "2026-08-15 09:42",
    owner: "平台管理员",
    mergeModel: true,
    parameters: defaultParameters
  },
  {
    id: "ft-240814-02",
    name: "Qwen3 指令跟随基线",
    state: "completed",
    model: "Qwen/Qwen3-1.7B",
    dataset: "qwenruozhi_v1",
    method: "SFT / LoRA",
    tuningMode: "lora",
    progress: 100,
    currentStep: "训练与评估完成",
    loss: 0.892,
    createdAt: "2026-08-14 16:18",
    owner: "刘士博",
    mergeModel: false,
    parameters: { ...defaultParameters, epochs: 2 }
  },
  {
    id: "ft-240814-01",
    name: "Qwen3 冒烟测试",
    state: "failed",
    model: "Qwen/Qwen3-1.7B",
    dataset: "qwenruozhi_v1",
    method: "SFT / LoRA",
    tuningMode: "lora",
    progress: 12,
    currentStep: "加载训练数据失败",
    createdAt: "2026-08-14 10:05",
    owner: "平台管理员",
    mergeModel: false,
    parameters: { ...defaultParameters, maxSequenceLength: 4096 }
  },
  {
    id: "ft-240813-02",
    name: "Qwen3 参数对照组",
    state: "stopped",
    model: "Qwen/Qwen3-1.7B",
    dataset: "qwenruozhi_v1",
    method: "SFT / LoRA",
    tuningMode: "lora",
    progress: 38,
    currentStep: "由用户停止",
    loss: 1.466,
    createdAt: "2026-08-13 19:26",
    owner: "李悦",
    mergeModel: false,
    parameters: { ...defaultParameters, learningRate: "1e-4" }
  },
  {
    id: "ft-240813-01",
    name: "Qwen3 夜间训练",
    state: "queued",
    model: "Qwen/Qwen3-1.7B",
    dataset: "qwenruozhi_v1",
    method: "SFT / LoRA",
    tuningMode: "lora",
    progress: 0,
    currentStep: "等待 GPU 资源",
    createdAt: "2026-08-13 18:40",
    owner: "张敏",
    mergeModel: true,
    parameters: defaultParameters
  }
];
