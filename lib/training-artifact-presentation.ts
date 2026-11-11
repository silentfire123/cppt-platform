import type { TrainingArtifact, TrainingTask } from "./types";

export function trainingArtifactInfo(
  artifact: TrainingArtifact,
  task: Pick<TrainingTask, "selectionMetric">,
  duplicateLabel = false,
  isSelected = false,
) {
  switch (artifact.type) {
    case "manifest":
      return { icon: "J", label: "产物清单（JSON）", description: "文件路径、大小与校验值，不含模型权重" };
    case "metrics":
      return { icon: "J", label: "训练状态与指标（JSON）", description: "训练/验证记录与运行状态，不含模型权重" };
    case "merged":
      return { icon: "M", label: "合并模型", description: artifact.sourceAdapterVariant ? `完整模型权重，来源：${artifact.sourceAdapterVariant === "final" ? "最终" : "最佳"} Adapter` : "完整模型权重，可独立加载" };
    case "adapter":
      return {
        icon: "A",
        label: duplicateLabel ? `${task.selectionMetric === "train_loss" ? "训练 Loss " : ""}最佳 = 最终 LoRA Adapter` : artifact.variant === "best" ? (task.selectionMetric === "train_loss" ? "训练 Loss 最佳 LoRA Adapter" : "最佳 LoRA Adapter") : "最终 LoRA Adapter",
        description: `${typeof artifact.checkpointStep === "number" ? `Step ${artifact.checkpointStep} · ` : ""}${typeof artifact.loss === "number" ? `${artifact.metric === "train_loss" ? "训练" : "验证"} Loss ${artifact.loss.toFixed(4)} · ` : typeof artifact.evalLoss === "number" ? `验证 Loss ${artifact.evalLoss.toFixed(4)} · ` : ""}${duplicateLabel ? "同一目录，仅展示一次" : isSelected ? "默认加载" : "可用于对照"}`,
      };
    default:
      return { icon: "J", label: "辅助产物", description: "训练附属信息，非模型权重" };
  }
}
