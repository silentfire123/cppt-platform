import type { TrainingTask } from "./types";

/** Merge an in-flight detail response without discarding a live log snapshot. */
export function mergeTrainingDetail(
  current: TrainingTask | null,
  fetched: TrainingTask,
  preserveLiveLogs: boolean,
): TrainingTask {
  if (!current || current.id !== fetched.id || !preserveLiveLogs) return fetched;
  return {
    ...fetched,
    logs: current.logs?.length ? current.logs : fetched.logs,
    lossHistory: current.lossHistory?.length ? current.lossHistory : fetched.lossHistory,
  };
}
