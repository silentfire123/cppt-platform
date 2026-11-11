export type YouyunInferenceImage = {
  id: string;
  name: string;
  tags: string[];
  description?: string;
  compatibility?: {
    version?: string | null;
    supportedGpu?: string[];
    recommendation?: "general" | "rtx5090" | string;
    recommendationReason?: string;
  };
};

export type YouyunInferenceImageAdvice = {
  kind: "general" | "rtx5090";
  badge: string;
  message: string;
  tone: "recommended" | "warning" | "neutral";
  compatible: boolean;
  recommended: boolean;
};

function searchableImageText(image: YouyunInferenceImage) {
  return [image.name, ...image.tags, image.description ?? ""]
    .join(" ")
    .toLowerCase();
}

function isRtx5090(gpuType: string) {
  return gpuType.replace(/[\s_-]/g, "").toLowerCase().includes("5090");
}

function isV100(gpuType: string) {
  return gpuType.replace(/[\s_-]/g, "").toLowerCase().includes("v100");
}

export function classifyYouyunInferenceImage(image: YouyunInferenceImage) {
  // Trust the backend's conclusion, which itself follows the provider catalog.
  // The 5090 gate is the only image-level constraint the platform asserts; no
  // per-version acceptance table is maintained here.
  const structuredKind = image.compatibility?.recommendation;
  if (structuredKind === "general" || structuredKind === "rtx5090") return structuredKind;
  if (searchableImageText(image).replace(/\s/g, "").includes("5090")) {
    return "rtx5090" as const;
  }
  return "general" as const;
}

export function getYouyunInferenceImageAdvice(
  image: YouyunInferenceImage,
  gpuType: string,
): YouyunInferenceImageAdvice {
  const kind = classifyYouyunInferenceImage(image);
  const gpuIs5090 = isRtx5090(gpuType);

  if (isV100(gpuType)) {
    return {
      kind,
      badge: "V100S 暂不支持",
      message: "当前优云官方 vLLM 镜像不包含 V100S（计算能力 7.0）所需的 CUDA 内核。",
      tone: "warning",
      compatible: false,
      recommended: false,
    };
  }

  if (kind === "rtx5090") {
    return gpuIs5090
      ? {
          kind,
          badge: "RTX 5090 推荐",
          message: "针对 RTX 5090 优化；当前 GPU 匹配，优先使用此镜像。",
          tone: "recommended",
          compatible: true,
          recommended: true,
        }
      : {
          kind,
          badge: "仅 RTX 5090",
          message: `该镜像仅用于 RTX 5090，与当前 ${gpuType || "GPU"} 不匹配。`,
          tone: "warning",
          compatible: false,
          recommended: false,
        };
  }

  const reason = image.compatibility?.recommendationReason;
  return {
    kind,
    badge: gpuIs5090 ? "官方镜像" : "默认推荐",
    message: reason || (gpuIs5090
      ? "优云官方 vLLM 镜像，可用于 RTX 5090；若存在 5090 专用镜像会优先推荐。"
      : `优云官方 vLLM 镜像，适合 ${gpuType || "大多数 GPU"}。`),
    tone: "recommended",
    compatible: true,
    recommended: !gpuIs5090,
  };
}

function versionSortKey(image: YouyunInferenceImage): number[] {
  const version = youyunInferenceImageVersion(image);
  const numbers = version?.match(/\d+/g);
  // Images without a provider version rank last; real versions are >= 0.
  if (!numbers?.length) return [-1];
  return numbers.slice(0, 4).map((part) => Number(part));
}

/** Order two images newest-first by the provider-reported version. */
export function compareYouyunInferenceImageVersion(
  a: YouyunInferenceImage,
  b: YouyunInferenceImage,
) {
  const left = versionSortKey(a);
  const right = versionSortKey(b);
  for (let index = 0; index < Math.max(left.length, right.length); index += 1) {
    const diff = (right[index] ?? 0) - (left[index] ?? 0);
    if (diff !== 0) return diff;
  }
  return a.name.localeCompare(b.name);
}

export function selectRecommendedYouyunInferenceImageId(
  images: YouyunInferenceImage[],
  gpuType: string,
  fallbackImageId?: string | null,
) {
  const gpuIs5090 = isRtx5090(gpuType);
  const usable = images.filter(
    (image) => getYouyunInferenceImageAdvice(image, gpuType).compatible,
  );
  const byKind = (kind: "general" | "rtx5090") =>
    usable
      .filter((image) => classifyYouyunInferenceImage(image) === kind)
      .sort(compareYouyunInferenceImageVersion)[0];

  // A 5090 prefers its dedicated image; every other GPU takes the newest
  // general-purpose image the provider currently publishes.
  const recommended = gpuIs5090
    ? byKind("rtx5090") ?? byKind("general")
    : byKind("general");
  if (recommended) return recommended.id;

  const fallback = usable.find((image) => image.id === fallbackImageId);
  if (fallback) return fallback.id;

  return usable[0]?.id ?? "";
}

export function youyunInferenceImageOptionLabel(
  image: YouyunInferenceImage,
  gpuType: string,
) {
  const advice = getYouyunInferenceImageAdvice(image, gpuType);
  return `${image.name}${image.tags.length ? ` · ${image.tags[0]}` : ""} · ${advice.badge}`;
}

export function youyunInferenceImageVersion(
  image: YouyunInferenceImage | null | undefined,
) {
  if (!image) return null;
  const compatibilityVersion = image.compatibility?.version?.trim();
  if (compatibilityVersion) return compatibilityVersion;
  for (const tag of image.tags) {
    if (/^v?\d+\.\d+(?:\.\d+)?(?:[-+][\w.-]+)?$/i.test(tag.trim())) {
      return tag.trim();
    }
  }
  const match = image.name.match(/vllm[^\d]{0,12}v(\d+\.\d+(?:\.\d+)?)/i);
  return match?.[1] ?? null;
}
