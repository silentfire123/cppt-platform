export type TrainingCreateRequest = {
  requestId: number;
  model: string;
  modelSource?: "public" | "modelscope";
  publicModelPath?: string;
  environment?: "local" | "youyun";
};

export type InferenceCreateRequest = {
  requestId: number;
  taskId: string;
  adapterVariant?: "best" | "final";
};

export type DeploymentArtifactChoice = {
  id: string;
  taskId: string;
  variant: "best" | "final";
  deployable: boolean;
};

export function selectDeploymentArtifactId(
  artifacts: DeploymentArtifactChoice[],
  request: Pick<InferenceCreateRequest, "taskId" | "adapterVariant">,
) {
  const taskArtifacts = artifacts.filter(
    (artifact) => artifact.taskId === request.taskId && artifact.deployable,
  );
  const requested = request.adapterVariant
    ? taskArtifacts.find((artifact) => artifact.variant === request.adapterVariant)
    : undefined;
  return (
    requested ??
    taskArtifacts.find((artifact) => artifact.variant === "best") ??
    taskArtifacts.find((artifact) => artifact.variant === "final") ??
    taskArtifacts[0] ??
    null
  )?.id ?? null;
}
