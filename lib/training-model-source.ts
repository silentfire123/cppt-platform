export type TrainingModelCatalog = "public" | "modelscope";

export function trainingModelCatalog({
  environment,
  selectedImageContainer,
  specsContainer,
}: {
  environment: "local" | "youyun";
  selectedImageContainer?: boolean | null;
  specsContainer?: boolean | null;
}): TrainingModelCatalog {
  if (environment !== "youyun") return "modelscope";
  const container = selectedImageContainer ?? specsContainer;
  // Youyun's default training image is a container image. While its catalog
  // metadata is still loading, keep the primary model picker on the public
  // model library. Only an explicitly confirmed VM image uses Repo-ID download.
  return container === false ? "modelscope" : "public";
}
