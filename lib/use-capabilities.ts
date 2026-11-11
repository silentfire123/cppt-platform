"use client";

import { useEffect, useState } from "react";
import { apiUrl } from "./api-config";

export type RuntimeCapabilities = {
  localRuntimeEnabled: boolean;
  cliTrainingEnabled: boolean;
  trainingProviders: string[];
  inferenceProviders: string[];
  modelSources: string[];
  evaluationEnabled: boolean;
  evaluationGenerationProvider: string;
  evaluationReusesTrainingInstance: boolean;
};

// Fail closed while loading or when the policy endpoint is unavailable.
export function useCapabilities() {
  const [capabilities, setCapabilities] =
    useState<RuntimeCapabilities | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    void fetch(apiUrl("/capabilities"), {
      credentials: "include",
      cache: "no-store",
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) throw new Error("读取运行能力失败");
        return response.json() as Promise<RuntimeCapabilities>;
      })
      .then(setCapabilities)
      .catch(() => {});
    return () => controller.abort();
  }, []);
  return {
    localRuntimeEnabled: capabilities?.localRuntimeEnabled === true,
    evaluationEnabled: capabilities?.evaluationEnabled === true,
    capabilities,
  };
}

export const LOCAL_RUNTIME_DISABLED = "生产环境已禁用本地运行";
