"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  Box,
  Boxes,
  CheckCircle2,
  Copy,
  Eye,
  Pencil,
  RefreshCw,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import { apiUrl } from "../lib/api-config";
import { formatApiError } from "../lib/api-error";
import type { InferenceCreateRequest } from "../lib/model-handoff";
import type { ModelLibraryView } from "../lib/interface-state";
import { Pagination } from "./pagination";
import { useDialog } from "./use-dialog";

async function modelRequest<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...options,
    credentials: "include",
    cache: "no-store",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(formatApiError(body, `模型接口请求失败（${response.status}）`));
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

type BaseModelStatus = "available" | "unavailable";

type SharedStorage = {
  id?: string;
  status:
    | "not_uploaded"
    | "pending_upload"
    | "uploading"
    | "available"
    | "failed"
    | "unavailable";
  sizeBytes?: number | null;
  manifestMd5?: string | null;
  bucket?: string | null;
  objectPrefix?: string | null;
  uri?: string | null;
  uploadedAt?: string | null;
  uploadedBy?: string | null;
  error?: string | null;
};

type PreviewFile = {
  path: string;
  sizeBytes?: number | null;
  md5?: string | null;
  status?: string | null;
};

type PreviewListing = {
  id: string;
  name: string;
  status: string;
  artifactType?: string | null;
  artifactVariant?: string | null;
  bucket?: string | null;
  objectPrefix?: string | null;
  files: PreviewFile[];
  source?: "local" | "minio";
  path?: string;
  truncated?: boolean;
};

type PreviewFileResult = PreviewFile & {
  kind: "text" | "binary" | "metadata";
  content?: string;
  message?: string;
  truncated?: boolean;
};

type BaseModel = {
  id: string;
  name: string;
  repo: string;
  revision: string;
  path: string;
  size: string;
  status: BaseModelStatus;
  source: "ModelScope 缓存" | "手工登记" | "优云公共模型库";
  publicModelPath?: string;
  publicModelId?: string;
  tag?: string;
  updatedAt: string;
  archived: boolean;
  sharedStorage?: SharedStorage;
};

type AdapterVariant = {
  key: "best" | "final";
  label: string;
  path: string;
  step: number | null;
  evalLoss: number | string | null;
  status: "available" | "unavailable";
  selected?: boolean;
  sharedStorage?: SharedStorage;
};

type TrainedModel = {
  id: string;
  name: string;
  baseModelId: string;
  baseModelName: string;
  taskId: string;
  method: "LoRA" | "QLoRA";
  dataset: string;
  updatedAt: string;
  size: string;
  archived: boolean;
  variants: AdapterVariant[];
};

type ModelAssetPage<T> = {
  items: T[];
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
};

type Detail =
  | { kind: "base"; item: BaseModel }
  | { kind: "trained"; item: TrainedModel }
  | null;

const initialBaseModels: BaseModel[] = [
  {
    id: "base-qwen25-3b",
    name: "Qwen2.5 3B Instruct",
    repo: "Qwen/Qwen2.5-3B-Instruct",
    revision: "master",
    path: "/root/.cache/modelscope/hub/Qwen/Qwen2.5-3B-Instruct",
    size: "6.4 GB",
    status: "available",
    source: "ModelScope 缓存",
    updatedAt: "2026-08-17 21:54",
    archived: false,
  },
  {
    id: "base-qwen25-7b",
    name: "Qwen2.5 7B Instruct",
    repo: "Qwen/Qwen2.5-7B-Instruct",
    revision: "master",
    path: "/root/.cache/modelscope/hub/Qwen/Qwen2.5-7B-Instruct",
    size: "14.8 GB",
    status: "available",
    source: "ModelScope 缓存",
    updatedAt: "2026-08-16 18:20",
    archived: false,
  },
  {
    id: "base-qwen3-17b",
    name: "Qwen3 1.7B",
    repo: "Qwen/Qwen3-1.7B",
    revision: "main",
    path: "/root/.cache/modelscope/hub/Qwen/Qwen3-1.7B",
    size: "3.6 GB",
    status: "available",
    source: "ModelScope 缓存",
    updatedAt: "2026-08-15 14:32",
    archived: false,
  },
  {
    id: "base-llama-31-8b",
    name: "Llama 3.1 8B Instruct",
    repo: "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    revision: "main",
    path: "D:/models/llama-3.1-8b-instruct",
    size: "—",
    status: "unavailable",
    source: "手工登记",
    updatedAt: "2026-08-12 09:40",
    archived: false,
  },
];

const initialTrainedModels: TrainedModel[] = [
  {
    id: "trained-ft-260815-215401",
    name: "客服知识问答 · Qwen2.5-3B",
    baseModelId: "base-qwen25-3b",
    baseModelName: "Qwen2.5 3B Instruct",
    taskId: "ft-260815-215401-88a5",
    method: "LoRA",
    dataset: "客服知识问答 v1",
    updatedAt: "2026-08-17 23:16",
    size: "182 MB",
    archived: false,
    variants: [
      {
        key: "best",
        label: "最佳 Adapter",
        path: "/root/xuntui-platform/artifacts/ft-260815-215401-88a5/adapter-best",
        step: 300,
        evalLoss: "1.102",
        status: "available",
      },
      {
        key: "final",
        label: "最终 Adapter",
        path: "/root/xuntui-platform/artifacts/ft-260815-215401-88a5/adapter-final",
        step: 909,
        evalLoss: "1.227",
        status: "available",
      },
    ],
  },
  {
    id: "trained-ft-260817-154620",
    name: "知识库助手 · Qwen3-1.7B",
    baseModelId: "base-qwen3-17b",
    baseModelName: "Qwen3 1.7B",
    taskId: "ft-260817-154620-e0e6",
    method: "QLoRA",
    dataset: "知识库对话 smoke",
    updatedAt: "2026-08-17 17:08",
    size: "96 MB",
    archived: false,
    variants: [
      {
        key: "best",
        label: "最佳 Adapter",
        path: "/root/xuntui-platform/artifacts/ft-260817-154620-e0e6/adapter-best",
        step: 3,
        evalLoss: "1.418",
        status: "available",
      },
      {
        key: "final",
        label: "最终 Adapter",
        path: "/root/xuntui-platform/artifacts/ft-260817-154620-e0e6/adapter-final",
        step: 4,
        evalLoss: "1.436",
        status: "available",
      },
    ],
  },
  {
    id: "trained-ft-260814-qa",
    name: "售后问答试验版",
    baseModelId: "base-qwen25-7b",
    baseModelName: "Qwen2.5 7B Instruct",
    taskId: "ft-260814-102210-4b2c",
    method: "QLoRA",
    dataset: "售后问答草稿",
    updatedAt: "2026-08-14 11:25",
    size: "210 MB",
    archived: true,
    variants: [
      {
        key: "best",
        label: "最佳 Adapter",
        path: "/root/xuntui-platform/artifacts/ft-260814-102210-4b2c/adapter-best",
        step: 180,
        evalLoss: "1.238",
        status: "available",
      },
      {
        key: "final",
        label: "最终 Adapter",
        path: "/root/xuntui-platform/artifacts/ft-260814-102210-4b2c/adapter-final",
        step: 240,
        evalLoss: "1.291",
        status: "available",
      },
    ],
  },
];

function statusLabel(status: BaseModelStatus) {
  return status === "available" ? "可用" : "路径不可用";
}

function storageLabel(storage?: SharedStorage) {
  if (!storage || storage.status === "not_uploaded") return "未上传";
  if (storage.status === "available") return "已备份";
  if (storage.status === "pending_upload") return "等待上传";
  if (storage.status === "uploading") return "上传中";
  if (storage.status === "failed") return "上传失败";
  return "存储不可用";
}

function fileStatusLabel(status?: string | null) {
  if (status === "available") return "可用";
  if (status === "uploading") return "上传中";
  if (status === "pending_upload") return "等待上传";
  if (status === "failed") return "上传失败";
  if (status === "unavailable") return "不可用";
  return status || "可读取";
}

function formatBytes(value?: number | null) {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = value;
  let index = -1;
  do {
    size /= 1024;
    index += 1;
  } while (size >= 1024 && index < units.length - 1);
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[index]}`;
}

function variantFor(model: TrainedModel, key: "best" | "final") {
  return model.variants.find((variant) => variant.key === key);
}

function localAvailable(variant: AdapterVariant) {
  return variant.status === "available" && Boolean(variant.path.trim());
}

function canDeploy(variant: AdapterVariant) {
  return localAvailable(variant) || variant.sharedStorage?.status === "available";
}

function deploymentLabel(variant: AdapterVariant) {
  return localAvailable(variant) ? "本地可部署" : variant.sharedStorage?.status === "available" ? "可从 MinIO 恢复部署" : "不可部署";
}

function defaultVariantFor(model: TrainedModel) {
  const deployableVariants = model.variants.filter(
    canDeploy,
  );
  return (
    deployableVariants.find((variant) => variant.selected) ??
    deployableVariants.find((variant) => variant.key === "best") ??
    deployableVariants.find((variant) => variant.key === "final") ??
    deployableVariants[0]
  );
}

function SharedStorageDetails({
  storage,
  onPreview,
}: {
  storage?: SharedStorage;
  onPreview?: () => void;
}) {
  return (
    <div className="drawer-section">
      <div className="section-title-row">
        <h3>共享存储</h3>
        <span
          className={`availability ${storage?.status === "available" ? "available" : "unavailable"}`}
        >
          <i />
          {storageLabel(storage)}
        </span>
      </div>
      <div className="drawer-info-grid">
        <div className="storage-address-cell">
          <span>MinIO 地址</span>
          <strong>{storage?.uri ?? "—"}</strong>
          {onPreview && (
            <button
              className="link-btn storage-preview-btn"
              disabled={storage?.status !== "available" || !storage.id}
              onClick={onPreview}
            >
              预览文件
            </button>
          )}
        </div>
        <div>
          <span>文件总大小</span>
          <strong>{formatBytes(storage?.sizeBytes)}</strong>
        </div>
        <div>
          <span>Manifest MD5</span>
          <strong>{storage?.manifestMd5 ?? "—"}</strong>
        </div>
        <div>
          <span>上传人 / 时间</span>
          <strong>
            {storage?.uploadedBy ?? "—"} · {storage?.uploadedAt ?? "—"}
          </strong>
        </div>
      </div>
      {storage?.error && <p className="drawer-help">{storage.error}</p>}
    </div>
  );
}

export function ModelManagementModule({
  view: tab,
  onOpenTraining,
  onOpenInference,
}: {
  view: ModelLibraryView;
  onOpenTraining?: (
    model: string,
    options?: { modelSource?: "public" | "modelscope"; publicModelPath?: string; environment?: "local" | "youyun" },
  ) => void;
  onOpenInference?: (
    request: Omit<InferenceCreateRequest, "requestId">,
  ) => void;
}) {
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([]);
  const [archivedTrainedModels, setArchivedTrainedModels] = useState<TrainedModel[]>([]);
  const [trainedPage, setTrainedPage] = useState(1);
  const [publicBaseModels, setPublicBaseModels] = useState<BaseModel[]>([]);
  const [publicBasePage, setPublicBasePage] = useState(1);
  const [publicBasePageInfo, setPublicBasePageInfo] = useState({
    pageSize: 5,
    total: 0,
    totalPages: 1,
  });
  const [publicBaseLoading, setPublicBaseLoading] = useState(false);
  const [publicBaseError, setPublicBaseError] = useState("");
  const [trainedPageInfo, setTrainedPageInfo] = useState({
    pageSize: 5,
    total: 0,
    totalPages: 1,
  });
  const [activeTrainedTotal, setActiveTrainedTotal] = useState(0);
  const [showArchived, setShowArchived] = useState(false);
  const [query, setQuery] = useState("");
  const [detail, setDetail] = useState<Detail>(null);
  const [showRegister, setShowRegister] = useState(false);
  const [scanState, setScanState] = useState<"idle" | "scanning">("idle");
  const [notice, setNotice] = useState("");
  const [editingName, setEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [registerForm, setRegisterForm] = useState({
    name: "",
    repo: "",
    revision: "main",
    path: "",
  });
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [uploadingAssetId, setUploadingAssetId] = useState("");
  const [preview, setPreview] = useState<PreviewListing | null>(null);
  const [previewFile, setPreviewFile] = useState<PreviewFileResult | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const previewRequest = useRef(0);
  const publicRequestKey = useRef("");
  const trainedRequestKey = useRef("");
  const archivedLoaded = useRef(false);
  const closePreview = useCallback(() => {
    previewRequest.current += 1;
    setPreview(null);
    setPreviewFile(null);
    setPreviewError("");
    setPreviewLoading(false);
  }, []);

  const detailDialogRef = useDialog<HTMLDivElement>(Boolean(detail), useCallback(() => setDetail(null), []));
  const registerDialogRef = useDialog<HTMLFormElement>(showRegister, useCallback(() => setShowRegister(false), []));
  const previewDialogRef = useDialog<HTMLDivElement>(Boolean(preview), closePreview);

  const publicBaseCount = publicBasePageInfo.total;
  const archivedCount = archivedTrainedModels.length;
  const matches = (value: string) =>
    value.toLowerCase().includes(query.trim().toLowerCase());
  const visibleTrainedModels = useMemo(
    () => (tab === "base" ? [] : trainedModels),
    [trainedModels, tab],
  );
  const visibleArchivedTrainedModels = useMemo(
    () =>
      archivedTrainedModels.filter((item) =>
        matches(
          `${item.name} ${item.baseModelName} ${item.taskId} ${item.dataset}`,
        ),
      ),
    [archivedTrainedModels, query],
  );

  function flash(message: string) {
    setNotice(message);
    window.setTimeout(() => setNotice(""), 2600);
  }
  function applyTrainedPage(
    result: ModelAssetPage<TrainedModel>,
    page: number,
  ) {
    setTrainedPageInfo({
      pageSize: result.pageSize,
      total: result.total,
      totalPages: result.totalPages,
    });
    if (page > result.totalPages) {
      setTrainedPage(result.totalPages);
      setTrainedModels([]);
    } else {
      setTrainedModels(result.items);
    }
  }
  function trainedPagePath(page: number, search: string) {
    const params = new URLSearchParams({
      kind: "trained",
      page: String(page),
      pageSize: "5",
    });
    if (search.trim()) params.set("q", search.trim());
    return `/model-assets?${params.toString()}`;
  }
  async function refreshArchivedTrainedAssets() {
    try {
      const nextArchivedTrainedModels = await modelRequest<TrainedModel[]>(
        "/model-assets?kind=trained&archivedOnly=true",
      );
      setArchivedTrainedModels(nextArchivedTrainedModels);
      archivedLoaded.current = true;
    } catch (error) {
      archivedLoaded.current = false;
      setLoadError(
        error instanceof Error ? error.message : "读取模型资产失败。",
      );
    }
  }
  async function refreshPublicBaseModels(page = publicBasePage, search = query) {
    const requestKey = `${page}:${search.trim()}`;
    publicRequestKey.current = requestKey;
    setPublicBaseLoading(true);
    try {
      const params = new URLSearchParams({ page: String(page), pageSize: "5" });
      if (search.trim()) params.set("q", search.trim());
      const result = await modelRequest<ModelAssetPage<BaseModel>>(
        "/models/public?" + params.toString(),
      );
      setPublicBaseModels(result.items);
      setPublicBasePageInfo({ pageSize: result.pageSize, total: result.total, totalPages: result.totalPages });
      setPublicBaseError("");
      if (page > result.totalPages) setPublicBasePage(result.totalPages);
    } catch (error) {
      if (publicRequestKey.current === requestKey) publicRequestKey.current = "";
      setPublicBaseModels([]);
      setPublicBaseError(error instanceof Error ? error.message : "读取优云公共模型库失败。");
    } finally {
      setPublicBaseLoading(false);
    }
  }
  async function refreshTrainedAssets(page = trainedPage, search = query) {
    const requestKey = `${page}:${search.trim()}`;
    trainedRequestKey.current = requestKey;
    setLoading(true);
    try {
      const result = await modelRequest<ModelAssetPage<TrainedModel>>(
        trainedPagePath(page, search),
      );
      applyTrainedPage(result, page);
      if (!search.trim()) setActiveTrainedTotal(result.total);
      setLoadError("");
    } catch (error) {
      if (trainedRequestKey.current === requestKey) trainedRequestKey.current = "";
      setLoadError(
        error instanceof Error ? error.message : "读取模型资产失败。",
      );
    } finally {
      setLoading(false);
    }
  }
  async function refreshAssets(page = trainedPage, search = query) {
    setLoading(true);
    try {
      const [
        nextTrainedPage,
        nextArchivedTrainedModels,
        activeTrainedCount,
      ] =
        await Promise.all([
          modelRequest<ModelAssetPage<TrainedModel>>(
            trainedPagePath(page, search),
          ),
          modelRequest<TrainedModel[]>(
            "/model-assets?kind=trained&archivedOnly=true",
          ),
          search.trim()
            ? modelRequest<ModelAssetPage<TrainedModel>>(
                "/model-assets?kind=trained&page=1&pageSize=1",
              )
            : Promise.resolve(null),
        ]);
      setArchivedTrainedModels(nextArchivedTrainedModels);
      trainedRequestKey.current = `${page}:${search.trim()}`;
      archivedLoaded.current = true;
      applyTrainedPage(nextTrainedPage, page);
      setDetail((current) => {
        if (current?.kind !== "trained") return current;
        const updated = [...nextTrainedPage.items, ...nextArchivedTrainedModels].find((item) => item.id === current.item.id);
        return updated ? { kind: "trained", item: updated } : current;
      });
      setActiveTrainedTotal(
        activeTrainedCount?.total ?? nextTrainedPage.total,
      );
      setLoadError("");
    } catch (error) {
      setLoadError(
        error instanceof Error ? error.message : "读取模型资产失败。",
      );
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    if (tab !== "trained") return;
    if (trainedRequestKey.current === `${trainedPage}:${query.trim()}`) return;
    void refreshTrainedAssets();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, trainedPage, query]);
  useEffect(() => {
    if (tab !== "trained") return;
    if (archivedLoaded.current) return;
    void refreshArchivedTrainedAssets();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);
  useEffect(() => {
    if (tab !== "base") return;
    if (publicRequestKey.current === `${publicBasePage}:${query.trim()}`) return;
    void refreshPublicBaseModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, publicBasePage, query]);
  const pendingUploads = [...trainedModels, ...archivedTrainedModels, ...(detail?.kind === "trained" ? [detail.item] : [])].some((item) =>
    item.variants.some((variant) => variant.sharedStorage?.status === "uploading" || variant.sharedStorage?.status === "pending_upload"),
  );
  useEffect(() => {
    if (!pendingUploads || tab !== "trained") return;
    const timer = window.setInterval(() => { void refreshAssets(); }, 3000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingUploads, tab, trainedPage, query]);
  function openDetail(next: Detail) {
    setDetail(next);
    setEditingName(false);
    setNameDraft(next?.item.name ?? "");
  }
  async function openPreview(storage?: SharedStorage) {
    if (!storage?.id) return;
    const request = ++previewRequest.current;
    setPreview({ id: storage.id, name: "MinIO 权重", status: storage.status, source: "minio", bucket: storage.bucket, objectPrefix: storage.objectPrefix, files: [] });
    setPreviewLoading(true);
    setPreviewError("");
    setPreviewFile(null);
    try {
      const result = await modelRequest<PreviewListing>(
        `/model-files/${encodeURIComponent(storage.id)}/preview`,
      );
      if (request !== previewRequest.current) return;
      setPreview({ ...result, source: "minio" });
    } catch (error) {
      if (request !== previewRequest.current) return;
      const message = error instanceof Error ? error.message : "读取模型文件失败。";
      setPreviewError(message);
      flash(message);
    } finally {
      if (request === previewRequest.current) setPreviewLoading(false);
    }
  }
  async function openLocalPreview(variant: AdapterVariant) {
    const request = ++previewRequest.current;
    setPreview({ id: variant.key, name: variant.label, status: variant.status, source: "local", path: variant.path, files: [] });
    setPreviewLoading(true);
    setPreviewError("");
    setPreviewFile(null);
    try {
      const result = await modelRequest<PreviewListing>("/artifacts/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: variant.path }),
      });
      if (request === previewRequest.current) setPreview({ ...result, id: variant.key, status: variant.status, source: "local" });
    } catch (error) {
      if (request === previewRequest.current) setPreviewError(error instanceof Error ? error.message : "读取服务器权重失败。");
    } finally {
      if (request === previewRequest.current) setPreviewLoading(false);
    }
  }
  async function openPreviewFile(file: PreviewFile) {
    if (!preview) return;
    const request = ++previewRequest.current;
    if (preview.source === "minio" && preview.status !== "available") {
      setPreviewFile({ ...file, kind: "metadata" });
      setPreviewLoading(false);
      setPreviewError("");
      return;
    }
    setPreviewLoading(true);
    setPreviewError("");
    try {
      const result = preview.source === "local"
        ? await modelRequest<PreviewFileResult>("/artifacts/preview", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: preview.path, relativePath: file.path }),
        })
        : await modelRequest<PreviewFileResult>(`/model-files/${encodeURIComponent(preview.id)}/preview/file?path=${encodeURIComponent(file.path)}`);
      if (request === previewRequest.current) setPreviewFile(result);
    } catch (error) {
      if (request === previewRequest.current) setPreviewError(error instanceof Error ? error.message : "读取文件内容失败。");
    } finally {
      if (request === previewRequest.current) setPreviewLoading(false);
    }
  }
  async function copyPath(path: string) {
    try {
      await navigator.clipboard.writeText(path);
      flash("路径已复制");
    } catch {
      flash("复制失败，请检查浏览器剪贴板权限。");
    }
  }
  async function uploadVariant(item: TrainedModel, variant: AdapterVariant) {
    if (uploadingAssetId) return;
    setUploadingAssetId(`${item.id}:${variant.key}`);
    try {
      const result = await modelRequest<{ sharedStorage: SharedStorage }>(`/model-assets/${encodeURIComponent(item.id)}/variants/${variant.key}/upload`, { method: "POST" });
      setDetail((current) => current?.kind === "trained" && current.item.id === item.id ? {
        kind: "trained", item: { ...current.item, variants: current.item.variants.map((entry) => entry.key === variant.key ? { ...entry, sharedStorage: result.sharedStorage } : entry) },
      } : current);
      await refreshAssets();
      flash(result.sharedStorage.status === "available" ? "权重已备份到 MinIO" : "已提交上传");
    } catch (error) {
      flash(error instanceof Error ? error.message : "上传权重失败。");
      await refreshAssets();
    } finally {
      setUploadingAssetId("");
    }
  }
  async function saveName() {
    if (!detail || !nameDraft.trim()) return;
    try {
      const updated = await modelRequest<BaseModel | TrainedModel>(
        `/model-assets/${detail.item.id}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: nameDraft.trim() }),
        },
      );
      setDetail(
        updated && "taskId" in updated
          ? { kind: "trained", item: updated }
          : { kind: "base", item: updated as BaseModel },
      );
      await refreshAssets();
      setEditingName(false);
      flash("显示名称已更新");
    } catch (error) {
      flash(error instanceof Error ? error.message : "更新模型名称失败。");
    }
  }
  async function toggleArchive(
    kind: "base" | "trained",
    id: string,
    archived: boolean,
  ) {
    try {
      const updated = await modelRequest<BaseModel | TrainedModel>(
        `/model-assets/${id}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ archived }),
        },
      );
      setDetail(
        updated && "taskId" in updated
          ? { kind: "trained", item: updated }
          : { kind: "base", item: updated as BaseModel },
      );
      await refreshAssets();
      flash(
        archived ? "已移入回收站，训练和部署选择器将不再展示" : "模型已恢复",
      );
    } catch (error) {
      flash(error instanceof Error ? error.message : "更新模型状态失败。");
    }
  }
  async function handleRegister(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (
      !registerForm.name.trim() ||
      !registerForm.repo.trim() ||
      !registerForm.path.trim()
    )
      return;
    try {
      await modelRequest<BaseModel>("/model-assets/base", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: registerForm.name.trim(),
          repo: registerForm.repo.trim(),
          revision: registerForm.revision.trim() || "main",
          path: registerForm.path.trim(),
        }),
      });
      await refreshAssets();
      setShowRegister(false);
      setRegisterForm({ name: "", repo: "", revision: "main", path: "" });
      flash("基模已登记，可用于新建训练");
    } catch (error) {
      flash(error instanceof Error ? error.message : "登记基模失败。");
    }
  }
  async function refreshAsset(assetId: string) {
    try {
      const updated = await modelRequest<BaseModel | TrainedModel>(
        `/model-assets/${assetId}/refresh`,
        { method: "POST" },
      );
      setDetail(
        updated && "taskId" in updated
          ? { kind: "trained", item: updated }
          : { kind: "base", item: updated as BaseModel },
      );
      await refreshAssets();
      flash("模型路径状态已刷新");
    } catch (error) {
      flash(error instanceof Error ? error.message : "刷新模型状态失败。");
    }
  }
  async function scanCache() {
    if (scanState === "scanning") return;
    setScanState("scanning");
    try {
      const result = await modelRequest<{ discovered: number }>(
        "/model-assets/scan",
        { method: "POST" },
      );
      await refreshAssets();
      flash(`扫描完成：发现 ${result.discovered} 个可识别仓库`);
    } catch (error) {
      flash(error instanceof Error ? error.message : "扫描本地缓存失败。");
    } finally {
      setScanState("idle");
    }
  }

  async function uploadBaseAsset(item: BaseModel) {
    if (uploadingAssetId) return;
    setUploadingAssetId(item.id);
    try {
      const result = await modelRequest<{ sharedStorage: SharedStorage }>(
        `/model-assets/${item.id}/upload`,
        { method: "POST" },
      );
      const updated = { ...item, sharedStorage: result.sharedStorage };
      setDetail({ kind: "base", item: updated });
      await refreshAssets();
      flash("基模文件已上传 MinIO，元数据已写入 PostgreSQL");
    } catch (error) {
      flash(error instanceof Error ? error.message : "上传共享存储失败。");
    } finally {
      setUploadingAssetId("");
    }
  }

  async function selectAdapterVariant(
    item: TrainedModel,
    variant: "best" | "final",
  ) {
    try {
      const updated = await modelRequest<{
        artifacts?: Array<{
          type?: string;
          variant?: string;
          path?: string;
          checkpointStep?: number | null;
          evalLoss?: number | null;
          selected?: boolean;
        }>;
      }>(`/tasks/${item.taskId}/adapter-selection`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ variant }),
      });
      setDetail({
        kind: "trained",
        item: {
          ...item,
          variants: item.variants.map((entry) => ({
            ...entry,
            selected: entry.key === variant,
          })),
        },
      });
      await refreshAssets();
      flash(
        `${variant === "best" ? "最佳" : "最终"} Adapter 已设为默认部署入口`,
      );
    } catch (error) {
      flash(error instanceof Error ? error.message : "切换默认 Adapter 失败。");
    }
  }

  return (
    <section className="content model-page">
      <div className="model-page-head">
        <div>
          <h1>模型管理</h1>
        </div>
      </div>
      {tab === "trained" && loadError && (
        <div className="model-error-banner">
          <strong>模型服务暂不可用</strong>
          <span>{loadError}</span>
          <button className="link-btn" onClick={() => void refreshAssets()}>
            重试
          </button>
        </div>
      )}
      <div className="model-summary">
        <div className="model-stat">
          <span className="model-stat-icon indigo"><Boxes size={16} strokeWidth={1.8} /></span>
          <div>
            <strong>{tab === "base" ? publicBaseCount : activeTrainedTotal}</strong>
            <span>{tab === "base" ? "公共模型" : "训练权重"}</span>
          </div>
          <small>{tab === "base" ? "当前查询总数" : "未归档任务"}</small>
        </div>
        <div className="model-stat">
          <span className="model-stat-icon green"><Sparkles size={16} strokeWidth={1.8} /></span>
          <div>
            <strong>{tab === "base" ? publicBaseModels.filter((item) => item.status === "available").length : trainedModels.flatMap((item) => item.variants).filter(localAvailable).length}</strong>
            <span>{tab === "base" ? "当前页可用" : "当前页本地可部署"}</span>
          </div>
          <small>{tab === "base" ? "公共基模" : "Best / Final 版本"}</small>
        </div>
        <div className="model-stat">
          <span className="model-stat-icon blue"><Box size={16} strokeWidth={1.8} /></span>
          <div>
            <strong>{tab === "base" ? publicBaseModels.filter((item) => item.status !== "available").length : trainedModels.flatMap((item) => item.variants).filter((variant) => variant.sharedStorage?.status === "available").length}</strong>
            <span>{tab === "base" ? "当前页不可用" : "当前页已备份"}</span>
          </div>
          <small>{tab === "base" ? "公共基模" : "MinIO 完整上传"}</small>
        </div>
        <div className="model-stat">
          <span className="model-stat-icon amber"><RefreshCw size={16} strokeWidth={1.8} /></span>
          <div>
            <strong>{tab === "base" ? publicBaseModels.filter((item) => !item.size || item.size === "暂无数据" || item.size === "—").length : archivedCount}</strong>
            <span>{tab === "base" ? "当前页大小未知" : "已归档权重"}</span>
          </div>
          <small>{tab === "base" ? "优云未提供大小" : "回收站"}</small>
        </div>
      </div>
      <div className="model-toolbar">
        <div className="model-search-wrap">
          <Search size={15} strokeWidth={1.8} />
          <input
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setTrainedPage(1);
              setPublicBasePage(1);
            }}
            placeholder={tab === "base" ? "搜索模型名称或公共模型 ID" : "搜索权重名称、来源基模或训练任务"}
            aria-label={tab === "base" ? "搜索模型名称或公共模型 ID" : "搜索权重名称、来源基模或训练任务"}
          />
        </div>
        {tab === "trained" && <label className="model-archive-filter"><input type="checkbox" checked={showArchived} onChange={(event) => setShowArchived(event.target.checked)} />回收站</label>}
        <button
          className="secondary-btn"
          disabled={tab === "base" ? publicBaseLoading : loading}
          onClick={() => {
            if (tab === "base") void refreshPublicBaseModels();
            else void refreshAssets();
          }}
        >
          <RefreshCw size={14} strokeWidth={1.8} />刷新
        </button>
      </div>
      {tab === "base" && (
        <div className="model-section">
          <div className="model-section-head">
            <div className="model-section-title">
              <span className="model-section-icon base"><Box size={17} strokeWidth={1.8} /></span>
              <div>
                <h2>优云公共模型库</h2>
              </div>
            </div>
            <span className="model-section-count">{publicBasePageInfo.total} 个</span>
          </div>
          <div className="model-table-wrap">
            <table className="model-table">
              <thead><tr><th>模型</th><th>来源</th><th>容器内路径</th><th>大小</th><th>状态</th><th>更新时间</th><th>操作</th></tr></thead>
              <tbody>
                {publicBaseModels.map((item) => (
                  <tr key={item.id}>
                    <td><button className="model-name-btn" onClick={() => openDetail({ kind: "base", item })}><span className="model-avatar base"><Box size={15} strokeWidth={1.8} /></span><span><strong>{item.name}</strong><small>{item.repo} · {item.revision}</small></span></button></td>
                    <td><span className="source-tag cache">优云公共模型库</span></td>
                    <td><code className="path-text">{item.path}</code></td>
                    <td>{!item.size || item.size === "—" ? "暂无数据" : item.size}</td>
                    <td><span className={`availability ${item.status}`}><i />{statusLabel(item.status)}</span></td>
                    <td>{item.updatedAt}</td>
                    <td><div className="model-row-actions"><button className="link-btn" onClick={() => onOpenTraining?.(item.publicModelPath || item.path, { modelSource: "public", publicModelPath: item.publicModelPath || item.path, environment: "youyun" })}>用于训练</button><button className="link-btn" onClick={() => openDetail({ kind: "base", item })}>详情</button></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {publicBaseLoading && <div className="model-empty">正在读取优云公共模型库...</div>}
            {!publicBaseLoading && publicBaseError && <div className="model-empty">{publicBaseError}</div>}
            {!publicBaseLoading && !publicBaseError && publicBaseModels.length === 0 && <div className="model-empty">公共模型库暂无匹配模型</div>}
            <Pagination page={publicBasePage} totalPages={publicBasePageInfo.totalPages} total={publicBasePageInfo.total} unit="个公共基模" onPageChange={setPublicBasePage} />
          </div>
        </div>
      )}
      {tab === "trained" && !showArchived && (
        <div className="model-section">
          <div className="model-section-head">
            <div className="model-section-title">
              <span className="model-section-icon trained"><Sparkles size={17} strokeWidth={1.8} /></span>
              <div>
              <h2>训练权重</h2>
              </div>
            </div>
            <span className="model-section-count">{trainedPageInfo.total} 个</span>
          </div>
          <div className="model-table-wrap">
            <table className="model-table trained-table">
              <thead>
                <tr>
                  <th>训练权重</th>
                  <th>来源基模</th>
                  <th>训练方式</th>
                  <th>部署 / MinIO 备份</th>
                  <th>最佳 Eval Loss</th>
                  <th>更新时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {visibleTrainedModels.map((item) => {
                  const best = variantFor(item, "best");
                  const final = variantFor(item, "final");
                  return (
                    <tr key={item.id}>
                      <td>
                        <button
                          className="model-name-btn"
                          onClick={() => openDetail({ kind: "trained", item })}
                        >
                          <span className="model-avatar trained"><Sparkles size={15} strokeWidth={1.8} /></span>
                          <span>
                            <strong title={item.name}>{item.name}</strong>
                            <small title={`${item.dataset} · ${item.taskId}`}>
                              {item.dataset} · {item.taskId}
                            </small>
                          </span>
                        </button>
                      </td>
                      <td>
                        <span
                          className="inline-model-link"
                          title={item.baseModelName}
                        >
                          {item.baseModelName}
                        </span>
                      </td>
                      <td>
                        <span className="method-tag">{item.method}</span>
                      </td>
                      <td>
                        <div className="adapter-pills">
                          <span className="adapter-pill best">
                            最佳{" "}
                            {best ? `${deploymentLabel(best)} · ${storageLabel(best.sharedStorage)}` : "未保留"}
                          </span>
                          <span className="adapter-pill final">
                            最终{" "}
                            {final
                              ? `${deploymentLabel(final)} · ${storageLabel(final.sharedStorage)}`
                              : "未保留"}
                          </span>
                        </div>
                      </td>
                      <td>
                        <strong className="loss-value">
                          {best?.evalLoss ?? "—"}
                        </strong>
                        <small className="step-label">
                          {best?.step ? `step ${best.step}` : "历史任务未记录"}
                        </small>
                      </td>
                      <td>{item.updatedAt}</td>
                      <td>
                        <div className="model-row-actions">
                          <button
                            className="link-btn"
                            onClick={() =>
                              openDetail({ kind: "trained", item })
                            }
                          >
                            详情
                          </button>
                          <button
                            className="link-btn"
                            disabled={!defaultVariantFor(item)}
                            onClick={() => {
                              const variant = defaultVariantFor(item);
                              if (!variant) return;
                              onOpenInference?.({
                                taskId: item.taskId,
                                adapterVariant: variant.key,
                              });
                            }}
                          >
                            部署
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {visibleTrainedModels.length === 0 && (
              <div className="model-empty">{loading ? "正在读取训练权重..." : "没有匹配的训练权重"}</div>
            )}
            <Pagination
              page={trainedPage}
              totalPages={trainedPageInfo.totalPages}
              total={trainedPageInfo.total}
              unit="个训练权重"
              onPageChange={setTrainedPage}
            />
          </div>
        </div>
      )}
      {tab === "trained" && showArchived && (
        <div className="model-section">
          <div className="model-section-head">
            <div className="model-section-title">
              <span className="model-section-icon archived"><RefreshCw size={17} strokeWidth={1.8} /></span>
              <div>
              <h2>回收站</h2>
              <p>归档模型不会出现在训练和部署选择器中</p>
              </div>
            </div>
            <span className="model-section-count">
              {visibleArchivedTrainedModels.length} 个
            </span>
          </div>
          <div className="model-archive-list">
            {visibleArchivedTrainedModels.map((item) => (
              <div className="archive-row" key={item.id}>
                <div>
                  <span className="model-avatar trained"><Sparkles size={15} strokeWidth={1.8} /></span>
                  <strong>{item.name}</strong>
                  <small>训练模型 · 来源 {item.baseModelName}</small>
                </div>
                <button
                  className="secondary-btn"
                  onClick={() => toggleArchive("trained", item.id, false)}
                >
                  恢复
                </button>
              </div>
            ))}
            {visibleArchivedTrainedModels.length === 0 && (
              <div className="model-empty">回收站为空</div>
            )}
          </div>
        </div>
      )}
      {notice && (
        <div className="model-toast">
          <span><CheckCircle2 size={15} strokeWidth={1.8} /></span>
          {notice}
        </div>
      )}
      {detail && (
        <div
          className="model-drawer-layer"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setDetail(null);
          }}
        >
          <aside ref={detailDialogRef} className="model-drawer" role="dialog" aria-modal="true" aria-labelledby="model-drawer-title" tabIndex={-1}>
            <div className="model-drawer-head">
              <div>
                <span className="drawer-kicker">
                  {detail.kind === "base" ? "公共模型详情" : "训练权重详情"}
                </span>
                {editingName ? (
                  <div className="name-edit">
                    <input
                      value={nameDraft}
                      onChange={(event) => setNameDraft(event.target.value)}
                      autoFocus
                    />
                    <button
                      className="link-btn"
                      onClick={() => void saveName()}
                    >
                      保存
                    </button>
                  </div>
                ) : (
                  <div className="drawer-title-line">
                    <h2 id="model-drawer-title">{detail.item.name}</h2>
                    {!(detail.kind === "base" && detail.item.source === "优云公共模型库") && <button
                      className="icon-btn"
                      title="编辑显示名称"
                      onClick={() => setEditingName(true)}
                    ><Pencil size={14} strokeWidth={1.8} /></button>}
                  </div>
                )}
                <p>
                  {detail.kind === "base"
                    ? detail.item.repo
                    : `${detail.item.method} · ${detail.item.dataset}`}
                </p>
              </div>
              <button type="button" className="close-btn" aria-label="关闭模型详情" title="关闭模型详情" onClick={() => setDetail(null)}>
                <X size={18} strokeWidth={1.8} />
              </button>
            </div>
            {detail.kind === "base" ? (
              <>
                <div className="drawer-status-row">
                  <span className={`availability ${detail.item.status}`}>
                    <i />
                    {statusLabel(detail.item.status)}
                  </span>
                    <span className="source-tag cache">{detail.item.source}</span>
                </div>
                <div className="drawer-info-grid">
                  <div>
                    <span>{detail.item.source === "优云公共模型库" ? "公共模型 ID" : "ModelScope ID"}</span>
                    <strong>{detail.item.source === "优云公共模型库" ? detail.item.publicModelId || "暂无数据" : detail.item.repo}</strong>
                  </div>
                  <div>
                    <span>标签</span>
                    <strong>{detail.item.tag || detail.item.revision || "暂无数据"}</strong>
                  </div>
                  <div>
                    <span>文件大小</span>
                    <strong>{!detail.item.size || detail.item.size === "—" ? "暂无数据" : detail.item.size}</strong>
                  </div>
                  <div>
                    <span>更新时间（北京时间）</span>
                    <strong>{detail.item.updatedAt || "暂无数据"}</strong>
                  </div>
                </div>
                <div className="drawer-section">
                    <h3>{detail.item.source === "优云公共模型库" ? "容器内路径" : "本地路径"}</h3>
                  <div className="path-box">
                    <code>{detail.item.path}</code>
                    <button
                      className="copy-path-btn"
                      title="复制容器路径"
                      aria-label="复制容器路径"
                      onClick={() => void copyPath(detail.item.path)}
                    >
                      <Copy size={15} />
                    </button>
                  </div>
                </div>
                {detail.item.source !== "优云公共模型库" && <SharedStorageDetails storage={detail.item.sharedStorage} onPreview={() => void openPreview(detail.item.sharedStorage)} />}
                <div className="drawer-section">
                  <h3>可用操作</h3>
                  <div className="drawer-action-grid">
                    <button className="primary-btn" disabled={detail.item.status !== "available"} onClick={() => onOpenTraining?.(detail.item.publicModelPath || detail.item.repo, detail.item.source === "优云公共模型库" ? { modelSource: "public", publicModelPath: detail.item.publicModelPath || detail.item.path, environment: "youyun" } : { modelSource: "modelscope" })}>
                      用于新建训练
                    </button>
                    {detail.item.source !== "优云公共模型库" && <button
                      className="secondary-btn"
                      disabled={
                        uploadingAssetId === detail.item.id ||
                        detail.item.status !== "available"
                      }
                      onClick={() => void uploadBaseAsset(detail.item)}
                    >
                      {uploadingAssetId === detail.item.id
                        ? "正在上传"
                        : detail.item.sharedStorage?.status === "available"
                          ? "重新上传"
                          : "上传到 MinIO"}
                    </button>}
                    {detail.item.source !== "优云公共模型库" && <button
                      className="secondary-btn"
                      onClick={() => void refreshAsset(detail.item.id)}
                    >
                      重新检测
                    </button>}
                  </div>
                </div>
                {detail.item.source !== "优云公共模型库" && <div className="drawer-footer-action">
                  <button
                    className="danger-link"
                    onClick={() =>
                      void toggleArchive("base", detail.item.id, true)
                    }
                  >
                    移入回收站
                  </button>
                </div>}
              </>
            ) : (
              <>
                <div className="drawer-status-row">
                  <span className={`availability ${detail.item.variants.some(localAvailable) ? "available" : detail.item.variants.some(canDeploy) ? "recoverable" : "unavailable"}`}>
                    <i />
                    {detail.item.variants.some(localAvailable) ? "本地权重可部署" : detail.item.variants.some(canDeploy) ? "可恢复部署" : "权重不可部署"}
                  </span>
                  <span className="method-tag">{detail.item.method}</span>
                  <span className="source-tag cache">训练产物</span>
                </div>
                <div className="lineage-card">
                  <div className="lineage-node">
                    <span>来源基模</span>
                    <strong>
                      {detail.item.baseModelName}
                    </strong>
                  </div>
                  <b><ArrowRight size={14} strokeWidth={1.8} /></b>
                  <div className="lineage-node">
                    <span>训练任务</span>
                    <strong>{detail.item.taskId}</strong>
                    <small>{detail.item.dataset}</small>
                  </div>
                </div>
                <div className="drawer-section">
                  <div className="section-title-row">
                    <h3>Adapter 变体</h3>
                    <span className="selected-hint">
                      {!defaultVariantFor(detail.item) ? "暂无可部署版本" : defaultVariantFor(detail.item)?.key === "best"
                        ? "默认选择最佳"
                        : "默认选择最终"}
                    </span>
                  </div>
                  <div className="variant-list">
                    {detail.item.variants.map((variant) => {
                      const isDeployable = canDeploy(variant);
                      const isDefault =
                        defaultVariantFor(detail.item)?.key === variant.key;
                      return (
                        <div
                          className={`variant-row weight-variant-row ${isDefault ? "selected" : ""}`}
                          key={variant.key}
                        >
                          <div className="weight-variant-head">
                            <div>
                            <strong>{variant.label}</strong>
                            <small>
                              step {variant.step ?? "—"} · Eval Loss{" "}
                              {variant.evalLoss ?? "—"}
                            </small>
                            </div>
                            <span className={`availability ${localAvailable(variant) ? "available" : canDeploy(variant) ? "recoverable" : "unavailable"}`}><i />{deploymentLabel(variant)}</span>
                          </div>
                          <div className="weight-path-row">
                            <span>服务器完整权重路径</span>
                            <code>{variant.path || "本地路径缺失"}</code>
                            <div className="weight-path-actions">
                              <button className="link-btn" disabled={!localAvailable(variant)} onClick={() => void openLocalPreview(variant)}><Eye size={14} />预览服务器文件</button>
                              <button className="icon-btn" title="复制服务器路径" aria-label="复制服务器路径" disabled={!variant.path} onClick={() => void copyPath(variant.path)}><Copy size={14} /></button>
                            </div>
                          </div>
                          <div className="weight-path-row">
                            <span>MinIO 备份 · {storageLabel(variant.sharedStorage)}</span>
                            <code>{variant.sharedStorage?.uri || (variant.sharedStorage?.bucket && variant.sharedStorage?.objectPrefix ? `s3://${variant.sharedStorage.bucket}/${variant.sharedStorage.objectPrefix}` : "尚无 MinIO 路径")}</code>
                            <small>Bucket：{variant.sharedStorage?.bucket || "暂无数据"}</small>
                            <small>Object prefix：{variant.sharedStorage?.objectPrefix || "暂无数据"}</small>
                            {variant.sharedStorage?.error && <p className="weight-storage-error" role="status">{variant.sharedStorage.error}</p>}
                            <div className="weight-path-actions">
                              <button className="link-btn" disabled={!variant.sharedStorage?.id} onClick={() => void openPreview(variant.sharedStorage)}><Eye size={14} />预览 MinIO 文件</button>
                              <button className="icon-btn" title="复制 MinIO 路径" aria-label="复制 MinIO 路径" disabled={!variant.sharedStorage?.uri} onClick={() => void copyPath(variant.sharedStorage?.uri || "")}><Copy size={14} /></button>
                              {variant.sharedStorage?.status !== "available" && <button className="link-btn" disabled={!localAvailable(variant) || Boolean(uploadingAssetId) || variant.sharedStorage?.status === "uploading" || variant.sharedStorage?.status === "pending_upload"} onClick={() => void uploadVariant(detail.item, variant)}><RefreshCw size={14} />{uploadingAssetId === `${detail.item.id}:${variant.key}` ? "正在上传" : variant.sharedStorage?.status === "failed" ? "重试上传" : "上传到 MinIO"}</button>}
                            </div>
                          </div>
                          <div className="weight-variant-actions">
                          <button
                            className={
                              isDefault ? "selected-variant" : "secondary-btn"
                            }
                            disabled={!localAvailable(variant)}
                            onClick={() =>
                              void selectAdapterVariant(
                                detail.item,
                                variant.key,
                              )
                            }
                          >
                            {isDefault
                              ? "当前默认"
                              : localAvailable(variant)
                                ? "选择"
                                : "本地恢复后可设默认"}
                          </button>
                          <button className="link-btn" disabled={!isDeployable || detail.item.archived} onClick={() => onOpenInference?.({ taskId: detail.item.taskId, adapterVariant: variant.key })}>{localAvailable(variant) ? "部署此版本" : "恢复并部署"}</button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div className="drawer-info-grid">
                  <div>
                    <span>文件大小</span>
                    <strong>{detail.item.size}</strong>
                  </div>
                  <div>
                    <span>更新时间</span>
                    <strong>{detail.item.updatedAt}</strong>
                  </div>
                  <div>
                    <span>来源数据集</span>
                    <strong>{detail.item.dataset}</strong>
                  </div>
                  <div>
                    <span>训练方式</span>
                    <strong>{detail.item.method}</strong>
                  </div>
                </div>
                <div className="drawer-section">
                  <h3>可用操作</h3>
                  <div className="drawer-action-grid">
                    <button
                      className="primary-btn"
                      disabled={!defaultVariantFor(detail.item) || detail.item.archived}
                      onClick={() => {
                        const variant = defaultVariantFor(detail.item);
                        if (!variant) return;
                        onOpenInference?.({
                          taskId: detail.item.taskId,
                          adapterVariant: variant.key,
                        });
                      }}
                    >
                      用于部署
                    </button>
                    <button
                      className="secondary-btn"
                      disabled={!defaultVariantFor(detail.item)?.path}
                      onClick={() => {
                        const selected = defaultVariantFor(detail.item);
                        if (selected) {
                          void copyPath(selected.path);
                        }
                      }}
                    >
                      复制默认路径
                    </button>
                  </div>
                </div>
                <div className="drawer-footer-action">
                  <button
                    className="danger-link"
                    onClick={() =>
                      void toggleArchive("trained", detail.item.id, true)
                    }
                  >
                    移入回收站
                  </button>
                </div>
              </>
            )}
          </aside>
        </div>
      )}
      {showRegister && (
        <div className="model-modal-layer" role="presentation">
          <form ref={registerDialogRef} className="model-modal" role="dialog" aria-modal="true" aria-labelledby="register-model-title" tabIndex={-1} onSubmit={handleRegister}>
            <div className="model-modal-head">
              <div>
                <span className="drawer-kicker">新增模型资产</span>
                <h2 id="register-model-title">登记基模</h2>
                <p>登记本地已有目录，ModelScope 仓库 ID 必填。</p>
              </div>
              <button
                type="button"
                className="close-btn"
                aria-label="关闭登记模型"
                title="关闭登记模型"
                onClick={() => setShowRegister(false)}
              >
                <X size={18} strokeWidth={1.8} />
              </button>
            </div>
            <div className="register-form">
              <label>
                显示名称 <b>*</b>
                <input
                  required
                  value={registerForm.name}
                  onChange={(event) =>
                    setRegisterForm({
                      ...registerForm,
                      name: event.target.value,
                    })
                  }
                  placeholder="例如：Qwen2.5 14B Instruct"
                />
              </label>
              <label>
                ModelScope 仓库 ID <b>*</b>
                <input
                  required
                  value={registerForm.repo}
                  onChange={(event) =>
                    setRegisterForm({
                      ...registerForm,
                      repo: event.target.value,
                    })
                  }
                  placeholder="例如：Qwen/Qwen2.5-14B-Instruct"
                />
              </label>
              <div className="register-two-col">
                <label>
                  Revision
                  <input
                    value={registerForm.revision}
                    onChange={(event) =>
                      setRegisterForm({
                        ...registerForm,
                        revision: event.target.value,
                      })
                    }
                    placeholder="main"
                  />
                </label>
                <label>
                  本地目录 <b>*</b>
                  <input
                    required
                    value={registerForm.path}
                    onChange={(event) =>
                      setRegisterForm({
                        ...registerForm,
                        path: event.target.value,
                      })
                    }
                    placeholder="/root/.cache/modelscope/..."
                  />
                </label>
              </div>
            </div>
            <div className="register-tip">
              <span>i</span>
              <p>登记只保存模型元数据和路径，不会复制或上传模型文件。</p>
            </div>
            <div className="model-modal-actions">
              <button
                type="button"
                className="secondary-btn"
                onClick={() => setShowRegister(false)}
              >
                取消
              </button>
              <button type="submit" className="primary-btn">
                登记基模
              </button>
            </div>
          </form>
        </div>
      )}
      {preview && (
        <div
          className="model-modal-layer preview-layer"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              closePreview();
            }
          }}
        >
          <div ref={previewDialogRef} className="model-modal preview-modal" role="dialog" aria-modal="true" aria-labelledby="model-preview-title" tabIndex={-1}>
            <div className="model-modal-head">
              <div>
                <span className="drawer-kicker">{preview.source === "local" ? "服务器文件预览" : "MinIO 文件预览"}</span>
                <h2 id="model-preview-title">{preview.name}</h2>
                <p>
                  {preview.source === "local" ? preview.path : `${preview.bucket ?? "—"} / ${preview.objectPrefix ?? "—"}`}
                </p>
              </div>
              <button
                type="button"
                className="close-btn"
                aria-label="关闭文件预览"
                title="关闭文件预览"
                onClick={closePreview}
              >
                <X size={18} strokeWidth={1.8} />
              </button>
            </div>
            {previewError && <div className="form-error">{previewError}</div>}
            {previewLoading && !previewFile ? (
              <div className="model-empty">正在读取文件...</div>
            ) : previewFile ? (
              <div className="preview-content-wrap">
                <div className="preview-file-head">
                  <button className="link-btn" onClick={() => { previewRequest.current += 1; setPreviewFile(null); setPreviewLoading(false); }}>
                    ← 返回文件列表
                  </button>
                  <strong>{previewFile.path}</strong>
                </div>
                {previewFile.kind === "text" ? (
                  <>
                    {previewFile.truncated && (
                      <p className="drawer-help">文件较大，仅展示前 512 KB。</p>
                    )}
                    <pre className="preview-code">{previewFile.content}</pre>
                  </>
                ) : (
                  <div className="preview-binary">
                    <strong>{previewFile.kind === "metadata" ? "文件上传尚未完成" : "这是二进制权重或模型文件"}</strong>
                    <span>大小：{formatBytes(previewFile.sizeBytes)}</span>
                    <span>MD5：{previewFile.md5 ?? "—"}</span>
                    <span>状态：{fileStatusLabel(previewFile.status)}</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="preview-file-list">
                {preview.files.map((file) => (
                  <button
                    className="preview-file-row"
                    key={file.path}
                    onClick={() => void openPreviewFile(file)}
                  >
                    <span>
                      <strong>{file.path}</strong>
                      <small>MD5：{file.md5 ?? "—"} · 状态：{fileStatusLabel(file.status)}</small>
                    </span>
                    <em>{formatBytes(file.sizeBytes)}</em>
                  </button>
                ))}
                {preview.files.length === 0 && (
                  <div className="model-empty">目录中没有可预览文件</div>
                )}
                {preview.truncated && <p className="drawer-help">文件较多，仅展示前 200 项。</p>}
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
