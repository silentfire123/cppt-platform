"use client";

import { DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Plus, RefreshCw, X } from "lucide-react";
import { WorkspaceNavigation, type WorkspaceModule } from "../components/workspace-navigation";
import { InferenceModule } from "../components/inference-module";
import { ModelManagementModule } from "../components/model-management-module";
import { OverviewModule } from "../components/overview-module";
import { TrainingModule } from "../components/training-module";
import { EvaluationModule } from "../components/evaluation-module";
import { DatasetRepairPanel } from "../components/dataset-repair-panel";
import { BackButton } from "../components/back-button";
import { Pagination } from "../components/pagination";
import { useDialog } from "../components/use-dialog";
import { Dataset, DatasetPurpose, DatasetRecordItem, DatasetRecordsResponse, DatasetState, DatasetType } from "../lib/types";
import { apiUrl, AUTH_CALLBACK_URL } from "../lib/api-config";
import { formatApiError } from "../lib/api-error";
import { inspectTextDataset } from "../lib/dataset-validation";
import { shouldRestoreInterfaceState, restoreModelLibraryView, type ModelLibraryView } from "../lib/interface-state";
import { useCapabilities } from "../lib/use-capabilities";

const DATASET_PAGE_SIZE = 6;
import type {
  InferenceCreateRequest,
  TrainingCreateRequest,
} from "../lib/model-handoff";

type View = "list" | "create" | "preview";
type Module = WorkspaceModule;
type Modal = "publish" | "success" | "delete" | "leave-create" | "import" | null;
type SelectedFile = { name: string; size: number; file: File };
type DatasetImportSource = {
  key: string;
  name: string;
  sizeBytes: number;
  lastModified?: string | null;
  supported: boolean;
};

async function datasetRequest<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), { ...options, credentials: "include", cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(formatApiError(body, `请求失败（${response.status}）`));
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

const stateMeta: Record<DatasetState, { label: string; className: string }> = {
  failed: { label: "校验失败", className: "status-failed" },
  ready: { label: "待发布", className: "status-ready" },
  published: { label: "已发布", className: "status-published" }
};

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatImportTime(value?: string | null) {
  if (!value) return "时间未知";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "时间未知";
  const parts = new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  }).formatToParts(date);
  const part = (type: Intl.DateTimeFormatPartTypes) => parts.find((item) => item.type === type)?.value ?? "";
  return `${part("year")}-${part("month")}-${part("day")} ${part("hour")}:${part("minute")}:${part("second")}`;
}

function previewText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  return JSON.stringify(value, null, 2);
}

function roleLabel(role: string) {
  return role === "user" ? "User" : role === "assistant" ? "Assistant" : role === "system" ? "System" : role;
}

function messagesText(value: unknown): string {
  if (!Array.isArray(value)) return previewText(value);
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((item) => `${roleLabel(String(item.role ?? "message"))}: ${previewText(item.content)}`)
    .join("\n\n");
}

function dpoValueText(value: unknown): string {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const message = value as Record<string, unknown>;
    if (typeof message.content === "string") return message.content;
  }
  return previewText(value);
}

type PreviewBlock = { role: string; content: string; tone?: "chosen" | "rejected" };

function previewBlocks(item: DatasetRecordItem | undefined, structure?: string): PreviewBlock[] {
  if (!item) return [];
  const record = item.record && typeof item.record === "object" ? item.record : {};
  const blocks: PreviewBlock[] = [];
  if (Array.isArray(record.messages)) {
    (record.messages as Array<Record<string, unknown>>).forEach((message) => {
      if (message && typeof message === "object") {
        blocks.push({ role: String(message.role ?? "message"), content: previewText(message.content) });
      }
    });
  }
  if (structure === "DPO") {
    blocks.push({ role: "assistant", content: dpoValueText(record.chosen), tone: "chosen" });
    blocks.push({ role: "assistant", content: dpoValueText(record.rejected), tone: "rejected" });
  }
  if (!blocks.length) blocks.push({ role: "record", content: previewText(record) });
  return blocks;
}

function blockClassName(block: PreviewBlock) {
  if (block.tone) return `record-message tone-${block.tone}`;
  return `record-message role-${block.role.toLowerCase()}`;
}

function CopyDatasetId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(id);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };
  return <button type="button" className="copy-dataset-id" title="复制数据集 ID" onClick={() => void copy()}>{copied ? "已复制" : "复制 ID"}</button>;
}

function DatasetPreview({ dataset, onBack }: { dataset: Dataset; onBack: () => void }) {
  const [page, setPage] = useState(1);
  const [pageInput, setPageInput] = useState("1");
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<DatasetRecordsResponse>({ items: [], page: 1, pageSize: 10, total: 0, totalPages: 1 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    const params = new URLSearchParams({ page: String(page), pageSize: "10" });
    if (query.trim()) params.set("query", query.trim());
    void fetch(`${apiUrl(`/datasets/${dataset.id}/records`)}?${params.toString()}`, { credentials: "include", cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(formatApiError(body, `读取数据失败（${response.status}）`));
        }
        return response.json() as Promise<DatasetRecordsResponse>;
      })
      .then((payload) => {
        if (!active) return;
        setResult(payload);
        setPage(payload.page);
        setPageInput(String(payload.page));
      })
      .catch((reason) => {
        if (active) setError(reason instanceof Error ? reason.message : "读取数据失败。");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, [dataset.id, page, query]);

  useEffect(() => {
    setSelectedIndex(result.items.length ? result.items[0].index : null);
  }, [result.items]);

  const goToPage = (nextPage: number) => {
    const target = Math.min(Math.max(nextPage, 1), result.totalPages);
    setPage(target);
    setPageInput(String(target));
  };

  const jumpToPage = () => {
    const target = Number.parseInt(pageInput, 10);
    if (Number.isFinite(target)) goToPage(target);
  };

  const pageItems = () => {
    const total = result.totalPages;
    if (total <= 7) return Array.from({ length: total }, (_, index) => index + 1) as Array<number | "ellipsis">;
    const items: Array<number | "ellipsis"> = [1];
    if (page > 4) items.push("ellipsis");
    const start = Math.max(2, page - 1);
    const end = Math.min(total - 1, page + 1);
    for (let value = start; value <= end; value += 1) items.push(value);
    if (page < total - 3) items.push("ellipsis");
    items.push(total);
    return items;
  };

  const columns = dataset.structure === "DPO"
    ? ["user", "chosen", "rejected"]
    : dataset.structure === "SFT"
      ? ["system", "user", "assistant"]
      : ["record"];
  const columnLabels: Record<string, string> = { system: "System", user: "User", assistant: "Assistant", chosen: "Assistant-chosen", rejected: "Assistant-rejected", record: "数据内容" };

  const cellText = (item: DatasetRecordItem, column: string) => {
    const record = item.record && typeof item.record === "object" ? item.record : {};
    if (column === "system" || column === "user" || column === "assistant") {
      const messages = Array.isArray(record.messages) ? record.messages : [];
      return messages.filter((message) => message && typeof message === "object" && (message as Record<string, unknown>).role === column).map((message) => previewText((message as Record<string, unknown>).content)).join("\n\n");
    }
    if (column === "chosen" || column === "rejected") return dpoValueText(record[column]);
    return previewText(record);
  };

  const selected = result.items.find((item) => item.index === selectedIndex) ?? result.items[0];
  const detailBlocks = previewBlocks(selected, dataset.structure);

  return <section className="content dataset-preview-content">
    <div className="title-row preview-title-row"><div><BackButton label="返回数据集列表" onClick={onBack} /><h1>查看数据集：{dataset.name}</h1><div className="subtitle">{dataset.type}{dataset.structure ? ` / ${dataset.structure}` : ""} · 当前为只读预览</div></div></div>
    <section className="panel dataset-info-panel"><div className="dataset-info-heading"><h2>基本信息</h2><span className={`status ${stateMeta[dataset.state].className}`}><span className="status-dot" />{stateMeta[dataset.state].label}</span></div><div className="dataset-info-grid"><div><span>文件名</span><strong>{dataset.name}</strong></div><div><span>数据集 ID</span><div className="dataset-id-value"><code>{dataset.id}</code><CopyDatasetId id={dataset.id} /></div></div><div><span>分类</span><strong>{dataset.type}{dataset.structure ? ` / ${dataset.structure}` : ""}</strong></div><div><span>导入状态</span><strong className="success-text">成功</strong></div><div><span>数据量</span><strong>{dataset.count}</strong></div><div><span>创建人</span><strong>{dataset.owner}</strong></div><div><span>更新时间</span><strong>{dataset.updatedAt}</strong></div></div></section>
      <section className="panel dataset-record-panel"><div className="record-toolbar"><input className="record-search" value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} placeholder="搜索数据内容" aria-label="搜索数据内容" /><span className="record-total">共 {result.total.toLocaleString()} 条</span></div>{error && <div className="form-error">{error}</div>}{loading ? <div className="empty">正在读取数据...</div> : result.items.length === 0 ? <div className="empty">没有匹配的数据</div> : <div className="record-browser"><div className="record-list-pane"><table className="record-list-table"><thead><tr><th>#</th>{columns.map((column) => <th key={column}>{columnLabels[column]}</th>)}</tr></thead><tbody>{result.items.map((item) => <tr key={item.index} tabIndex={0} role="button" aria-label={`选择样本 ${item.index}`} className={selected && item.index === selected.index ? "selected" : ""} onClick={() => setSelectedIndex(item.index)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelectedIndex(item.index); } }}><td>{item.index}</td>{columns.map((column) => <td key={column}><div className="record-cell-text">{cellText(item, column) || "—"}</div></td>)}</tr>)}</tbody></table></div><div className="record-detail-pane"><div className="record-detail-head"><strong>样本 #{selected?.index ?? "—"}</strong><span>点击左侧列表切换样本</span></div><div className="record-detail-body">{detailBlocks.map((block, blockIndex) => <div key={blockIndex} className={blockClassName(block)}><span className="record-message-role">{block.tone === "chosen" ? "Assistant · Chosen ✓" : block.tone === "rejected" ? "Assistant · Rejected ✗" : roleLabel(block.role)}</span><p>{block.content || "—"}</p></div>)}</div></div></div>}
      <div className="record-pagination"><button type="button" className="page-btn" disabled={page <= 1} aria-label="上一页" title="上一页" onClick={() => goToPage(page - 1)}><ChevronLeft size={15} aria-hidden="true" /></button>{pageItems().map((item, index) => item === "ellipsis" ? <span className="page-ellipsis" key={`ellipsis-${index}`}>…</span> : <button type="button" className={`page-btn ${item === page ? "active" : ""}`} key={item} onClick={() => goToPage(item)}>{item}</button>)}<button type="button" className="page-btn" disabled={page >= result.totalPages} aria-label="下一页" title="下一页" onClick={() => goToPage(page + 1)}><ChevronRight size={15} aria-hidden="true" /></button><div className="page-jump"><span>跳转到</span><input value={pageInput} onChange={(event) => setPageInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") jumpToPage(); }} inputMode="numeric" aria-label="跳转页码" /><button type="button" className="secondary-btn" onClick={jumpToPage}>确定</button></div></div>
    </section>
  </section>;
}

export default function Home() {
  const { capabilities, evaluationEnabled } = useCapabilities();
  const handoffSequence = useRef(0);
  const [module, setModule] = useState<Module>("overview");
  const [modelView, setModelView] = useState<ModelLibraryView>("base");
  const [trainingCreateRequest, setTrainingCreateRequest] =
    useState<TrainingCreateRequest | null>(null);
  const [inferenceCreateRequest, setInferenceCreateRequest] =
    useState<InferenceCreateRequest | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetsLoaded, setDatasetsLoaded] = useState(false);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsError, setDatasetsError] = useState("");
  const [view, setView] = useState<View>("list");
  const [selected, setSelected] = useState<Dataset | null>(null);
  const [previewDataset, setPreviewDataset] = useState<Dataset | null>(null);
  const [modal, setModal] = useState<Modal>(null);
  const [query, setQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState("全部类型");
  const [stateFilter, setStateFilter] = useState("全部状态");
  const [name, setName] = useState("");
  const [datasetType, setDatasetType] = useState<DatasetType>("文本生成");
  const [datasetPurpose, setDatasetPurpose] = useState<DatasetPurpose>("training");
  const [structure, setStructure] = useState<"SFT" | "DPO">("SFT");
  const [files, setFiles] = useState<SelectedFile[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState("");
  const [fileError, setFileError] = useState("");
  const [formError, setFormError] = useState("");
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);
  const [validationRows, setValidationRows] = useState(0);
  const [interfaceRestored, setInterfaceRestored] = useState(false);
  const [restoredDatasetId, setRestoredDatasetId] = useState("");
  const [datasetPage, setDatasetPage] = useState(1);
  const [importSources, setImportSources] = useState<DatasetImportSource[]>([]);
  const [importSourceKey, setImportSourceKey] = useState("");
  const [importName, setImportName] = useState("");
  const [importStructure, setImportStructure] = useState<"SFT" | "DPO">("SFT");
  const [importLoading, setImportLoading] = useState(false);
  const [importSubmitting, setImportSubmitting] = useState(false);
  const [importError, setImportError] = useState("");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.has("code") || params.has("error")) {
      window.location.replace(`${AUTH_CALLBACK_URL}?${params.toString()}`);
      return;
    }
    const authError = params.get("auth_error");
    if (authError) {
      setFormError(authError);
      window.history.replaceState({}, document.title, window.location.pathname);
    }
    void fetch(apiUrl("/auth/me"), { credentials: "include", cache: "no-store" }).then((response) => {
      if (response.status === 401) window.location.assign(apiUrl("/auth/login"));
    }).catch(() => {
      // The backend health check will surface a real availability error later.
    });
  }, []);

  const closeDatasetOverlay = useCallback(() => {
    if (modal) setModal(null);
    else setSelected(null);
  }, [modal]);
  const datasetDrawerRef = useDialog<HTMLDivElement>(Boolean(selected && !modal), closeDatasetOverlay);
  const datasetModalRef = useDialog<HTMLDivElement>(Boolean(modal), closeDatasetOverlay);

  const refreshDatasets = useCallback(async () => {
    setDatasetsLoading(true);
    setDatasetsError("");
    try {
      setDatasets(await datasetRequest<Dataset[]>("/datasets"));
      setDatasetsLoaded(true);
    } catch (error) {
      const message = error instanceof Error ? error.message : "读取数据集失败。";
      setDatasetsLoaded(true);
      setDatasetsError(message);
      setFormError(message);
    } finally {
      setDatasetsLoading(false);
    }
  }, []);

  useEffect(() => {
    const navigationType = (window.performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined)?.type;
    const restoreState = shouldRestoreInterfaceState(navigationType);
    try {
      const saved = sessionStorage.getItem("xuntui-interface-state");
      if (restoreState && saved) {
        const state = JSON.parse(saved) as { module?: Module; view?: View; datasetId?: string; modelView?: unknown };
        setModelView(restoreModelLibraryView(state.modelView));
        if (state.module && ["overview", "datasets", "training", "evaluation", "models", "inference"].includes(state.module)) setModule(state.module);
        if (state.view && ["list", "create", "preview"].includes(state.view)) setView(state.view);
        if (state.datasetId) setRestoredDatasetId(state.datasetId);
      }
    } catch {
      // A stale or unavailable session state should not prevent the app from loading.
    } finally {
      setInterfaceRestored(true);
    }
  }, []);

  useEffect(() => {
    if (interfaceRestored && module === "datasets" && !datasetsLoaded && !datasetsLoading) {
      void refreshDatasets();
    }
  }, [datasetsLoaded, datasetsLoading, interfaceRestored, module, refreshDatasets]);

  useEffect(() => {
    if (!interfaceRestored) return;
    try {
      sessionStorage.setItem("xuntui-interface-state", JSON.stringify({ module, view, modelView, datasetId: previewDataset?.id ?? "" }));
    } catch {
      // Session storage is best effort only.
    }
  }, [interfaceRestored, module, modelView, previewDataset?.id, view]);

  useEffect(() => {
    if (view !== "preview" || !restoredDatasetId || !datasets.length) return;
    const restored = datasets.find((item) => item.id === restoredDatasetId);
    if (restored) setPreviewDataset(restored);
    else {
      setView("list");
      setRestoredDatasetId("");
    }
  }, [datasets, restoredDatasetId, view]);

  const filtered = useMemo(() => datasets.filter((item) => {
    const matchesName = item.name.includes(query) || item.subtitle.includes(query);
    const matchesType = typeFilter === "全部类型" || item.type === typeFilter;
    const matchesState = stateFilter === "全部状态" || item.state === stateFilter;
    return matchesName && matchesType && matchesState;
  }), [datasets, query, typeFilter, stateFilter]);

  const datasetTotalPages = Math.max(1, Math.ceil(filtered.length / DATASET_PAGE_SIZE));
  const datasetSafePage = Math.min(datasetPage, datasetTotalPages);
  const pagedDatasets = useMemo(
    () => filtered.slice((datasetSafePage - 1) * DATASET_PAGE_SIZE, datasetSafePage * DATASET_PAGE_SIZE),
    [filtered, datasetSafePage],
  );

  useEffect(() => {
    setDatasetPage(1);
  }, [query, typeFilter, stateFilter]);

  useEffect(() => {
    if (!capabilities || evaluationEnabled) return;
    if (module === "evaluation") setModule("overview");
    if (datasetPurpose === "evaluation") setDatasetPurpose("training");
  }, [capabilities, datasetPurpose, evaluationEnabled, module]);

  function resetCreateForm() {
    setName("");
    setDatasetType("文本生成");
    setDatasetPurpose("training");
    setStructure("SFT");
    setFiles([]);
    setUploadProgress(0);
    setUploading(false);
    setProcessing(false);
    setProcessingStage("");
    setFileError("");
    setFormError("");
    setValidationErrors([]);
    setValidationWarnings([]);
    setValidationRows(0);
  }

  function openCreate() {
    resetCreateForm();
    setSelected(null);
    setPreviewDataset(null);
    setView("create");
  }

  async function openImport() {
    setImportError("");
    setImportSources([]);
    setImportSourceKey("");
    setImportName("");
    setImportStructure("SFT");
    setModal("import");
    setImportLoading(true);
    try {
      const sources = await datasetRequest<DatasetImportSource[]>("/datasets/import-sources");
      setImportSources(sources);
      const firstSupported = sources.find((source) => source.supported);
      if (firstSupported) setImportSourceKey(firstSupported.key);
      if (!firstSupported) setImportError("exports/ 目录中没有可导入的 JSONL 文件。");
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "读取导入目录失败。");
    } finally {
      setImportLoading(false);
    }
  }

  function selectImportSource(key: string) {
    setImportSourceKey(key);
  }

  function hasCreateDraft() {
    return Boolean(name.trim() || files.length || validationErrors.length || validationWarnings.length);
  }

  function requestCreateBack() {
    if (processing || uploading) return;
    if (hasCreateDraft()) {
      setModal("leave-create");
      return;
    }
    setView("list");
  }

  function leaveCreate() {
    setModal(null);
    resetCreateForm();
    setView("list");
  }

  function openPreview(dataset: Dataset) {
    setSelected(null);
    setRestoredDatasetId(dataset.id);
    setPreviewDataset(dataset);
    setView("preview");
  }

  function changeDatasetType(type: DatasetType) {
    setDatasetType(type);
    setFiles([]);
    setUploadProgress(0);
    setUploading(false);
    setFileError("");
    setValidationErrors([]);
    setValidationWarnings([]);
    setValidationRows(0);
  }

  function changeDatasetPurpose(next: DatasetPurpose) {
    setDatasetPurpose(next);
    setFiles([]); setUploadProgress(0); setFileError(""); setValidationErrors([]); setValidationWarnings([]); setValidationRows(0);
    if (next === "evaluation") { setStructure("SFT"); setDatasetType("文本生成"); }
  }

  function changeStructure(next: "SFT" | "DPO") {
    setStructure(next);
    setFiles([]);
    setUploadProgress(0);
    setUploading(false);
    setFileError("");
    setValidationErrors([]);
    setValidationWarnings([]);
    setValidationRows(0);
  }

  function validateFiles(fileList: File[]) {
    const imageExtensions = ["jpg", "jpeg", "png", "webp"];
    const textExtensions = ["jsonl"];
    const allowed = datasetType === "文生图" ? imageExtensions : textExtensions;
    const invalidFormat = fileList.find((file) => !allowed.includes(file.name.split(".").pop()?.toLowerCase() ?? ""));
    if (invalidFormat) return `文件 ${invalidFormat.name} 格式不支持，请上传 ${allowed.join(" / ")} 文件。`;
    if (datasetType === "文生图" && fileList.length > 200) return "单个文生图数据集最多上传 200 张图片。";
    const oversizedImage = datasetType === "文生图" && fileList.find((file) => file.size > 20 * 1024 * 1024);
    if (oversizedImage) return `图片 ${oversizedImage.name} 超过 20MB。`;
    if (datasetPurpose === "evaluation" && fileList.length > 10) return "单个评测集最多上传 10 个 JSONL 文件。";
    if (datasetPurpose === "evaluation" && fileList.some((file) => file.size > 200 * 1024 * 1024)) return "评测集单个文件不能超过 200MB。";
    if (datasetType === "文本生成" && datasetPurpose !== "evaluation" && fileList.some((file) => file.size > 2 * 1024 * 1024 * 1024)) return "单个数据文件不能超过 2GB。";
    return "";
  }

  async function inspectTextFile(file: File) {
    const text = await file.text();
    const inspection = inspectTextDataset(text, structure);
    setValidationErrors(inspection.errors);
    setValidationWarnings(inspection.warnings);
    setValidationRows(inspection.rows);
  }

  async function beginUpload(fileList: File[]) {
    const error = validateFiles(fileList);
    if (error) {
      setFileError(error);
      return;
    }
    if (!fileList.length) return;
    setFileError("");
    setValidationErrors([]);
    setValidationWarnings([]);
    setValidationRows(0);
    setFiles(fileList.map((file) => ({ name: file.name, size: file.size, file })));
    if (datasetType === "文本生成" && datasetPurpose !== "evaluation" && fileList[0]) await inspectTextFile(fileList[0]);
    setUploadProgress(0);
  }

  function handleFileInput(fileList: FileList | null) {
    if (fileList) beginUpload(Array.from(fileList));
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    beginUpload(Array.from(event.dataTransfer.files));
  }

  function uploadTextDataset(file: SelectedFile, existingDatasetId?: string): Promise<Dataset> {
    return new Promise((resolve, reject) => {
      const query = new URLSearchParams({ name: name.trim(), structure, filename: file.name, datasetType: datasetPurpose });
      if (existingDatasetId) query.set("datasetId", existingDatasetId);
      const request = new XMLHttpRequest();
      request.open("POST", `${apiUrl('/datasets/upload')}?${query.toString()}`);
      request.setRequestHeader("Content-Type", "application/octet-stream");
      request.upload.onprogress = (event) => {
        if (event.lengthComputable) setUploadProgress(Math.round((event.loaded / event.total) * 100));
      };
      request.upload.onload = () => {
        setUploadProgress(100);
        setUploading(false);
        setProcessingStage("正在初始化文件并校验数据格式...");
      };
      request.onerror = () => reject(new Error("上传失败，请检查本地训练服务。"));
      request.onload = () => {
        let body: Record<string, unknown> = {};
        try { body = JSON.parse(request.responseText); } catch { /* Use the status fallback. */ }
        if (request.status >= 200 && request.status < 300) resolve(body as Dataset);
        else reject(new Error(formatApiError(body, `上传失败（${request.status}）`)));
      };
      request.send(file.file);
    });
  }

  async function createAndProcess() {
    if (!name.trim()) {
      setFormError("请填写数据集名称。 ");
      return;
    }
    if (!files.length) {
      setFormError("请先选择规定格式的文件。 ");
      return;
    }
    if (datasetType !== "文本生成") {
      setFormError("文生图数据集当前仅完成原型，真实上传将在视觉模型微调阶段接入。");
      return;
    }
    setUploading(true);
    setProcessing(true);
    setProcessingStage("正在上传数据集文件...");
    setFormError("");
    try {
      let dataset: Dataset | undefined;
      for (const file of files) dataset = await uploadTextDataset(file, dataset?.id);
      if (!dataset) throw new Error("没有可上传的数据文件。");
      setDatasets((items) => [dataset, ...items.filter((item) => item.id !== dataset.id)]);
      await refreshDatasets();
      setSelected(null);
      setView("list");
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "创建数据集失败。");
    } finally {
      setUploading(false);
      setProcessing(false);
      setProcessingStage("");
    }
  }

  function publish() {
    setModal("publish");
  }

  async function confirmPublish() {
    if (!selected) return;
    try {
      const published = await datasetRequest<Dataset>(`/datasets/${selected.id}/publish`, { method: "POST" });
      setDatasets((items) => items.map((item) => item.id === selected.id ? published : item));
      setSelected(published);
      setModal("success");
    } catch (error) {
      setModal(null);
      setFormError(error instanceof Error ? error.message : "发布数据集失败。");
    }
  }

  async function importDataset() {
    if (!importSourceKey) {
      setImportError("请选择要导入的数据集文件。");
      return;
    }
    if (!importName.trim()) {
      setImportError("请填写数据集名称。");
      return;
    }
    setImportSubmitting(true);
    setImportError("");
    try {
      const imported = await datasetRequest<Dataset>("/datasets/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sourceKey: importSourceKey, name: importName.trim(), structure: importStructure }),
      });
      setDatasets((items) => [imported, ...items.filter((item) => item.id !== imported.id)]);
      setModal(null);
      setSelected(null);
      setView("list");
      await refreshDatasets();
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "导入数据集失败。");
    } finally {
      setImportSubmitting(false);
    }
  }

  async function deleteDataset() {
    if (!selected) return;
    try {
      await datasetRequest(`/datasets/${selected.id}`, { method: "DELETE" });
      setDatasets((items) => items.filter((item) => item.id !== selected.id));
      setSelected(null);
      setModal(null);
    } catch (error) {
      setModal(null);
      setFormError(error instanceof Error ? error.message : "删除数据集失败。");
    }
  }

  function actionsFor(item: Dataset) {
    if (item.state === "failed") return ["错误详情", "只读", "删除"];
    if (item.state === "ready") return ["详细", "只读", "发布", "删除"];
    return ["详细", "只读", "删除"];
  }

  async function handleRowAction(item: Dataset, action: string) {
    if (["详细", "错误详情"].includes(action)) setSelected(item);
    if (action === "只读") openPreview(item);
    if (action === "发布") {
      setSelected(item);
      setModal("publish");
    }
    if (action === "删除") {
      setSelected(item);
      setModal("delete");
    }
  }

  const accept = datasetType === "文生图"
    ? ".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
    : ".jsonl,application/x-ndjson";

  function openModule(nextModule: Module) {
    if (nextModule === "evaluation" && !evaluationEnabled) return;
    setModule(nextModule);
    if (nextModule === "datasets") {
      setView("list");
      setPreviewDataset(null);
      setSelected(null);
    }
  }

  function openTrainingWithModel(
    model: string,
    options?: Pick<TrainingCreateRequest, "modelSource" | "publicModelPath" | "environment">,
  ) {
    handoffSequence.current += 1;
    setTrainingCreateRequest({ requestId: handoffSequence.current, model, ...options });
    setModule("training");
  }

  function openInferenceWithArtifact(
    request: Omit<InferenceCreateRequest, "requestId">,
  ) {
    handoffSequence.current += 1;
    setInferenceCreateRequest({
      requestId: handoffSequence.current,
      ...request,
    });
    setModule("inference");
  }

  return (
    <div className="app-shell">
      <WorkspaceNavigation activeModule={module} modelView={modelView} evaluationEnabled={evaluationEnabled} onNavigate={openModule} onModelView={(next) => { setModelView(next); openModule("models"); }} />

      <main className={`main ${module === "inference" ? "main-inference" : ""}`}>
        {module === "overview" && <OverviewModule onOpenModule={openModule} />}
        {module === "datasets" && view === "list" && (
          <section className="content">
            <div className="title-row">
              <div><h1>数据集</h1><div className="subtitle">管理用于文本生成与文生图微调的数据资产</div></div>
              <div className="title-actions"><button type="button" className="secondary-btn" onClick={() => void openImport()}>导入数据集</button><button type="button" className="primary-btn" onClick={openCreate}><Plus size={15} aria-hidden="true" />新建数据集</button></div>
            </div>
            <div className="summary">
              <div className="metric"><div className="metric-value">{datasets.length}</div><div className="metric-label">全部数据集</div></div>
              <div className="metric"><div className="metric-value red">{datasets.filter((item) => item.state === "failed").length}</div><div className="metric-label">校验失败</div></div>
              <div className="metric"><div className="metric-value blue">{datasets.filter((item) => item.state === "ready").length}</div><div className="metric-label">待发布</div></div>
              <div className="metric"><div className="metric-value green">{datasets.filter((item) => item.state === "published").length}</div><div className="metric-label">已发布</div></div>
            </div>
            <div className="filters">
              <input className="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索数据集名称" aria-label="搜索数据集名称" />
              <select className="filter-select" value={typeFilter} onInput={(event) => setTypeFilter(event.currentTarget.value)} onChange={(event) => setTypeFilter(event.target.value)} aria-label="任务类型筛选">
                <option>全部类型</option><option>文本生成</option><option>文生图</option>
              </select>
              <select className="filter-select" value={stateFilter} onInput={(event) => setStateFilter(event.currentTarget.value)} onChange={(event) => setStateFilter(event.target.value)} aria-label="状态筛选">
                <option value="全部状态">全部状态</option>
                <option value="failed">校验失败</option>
                <option value="ready">待发布</option>
                <option value="published">已发布</option>
              </select>
              <div className="filter-spacer" />
              <button type="button" className="secondary-btn refresh" title="刷新数据集" aria-label="刷新数据集" onClick={() => void refreshDatasets()}><RefreshCw size={15} aria-hidden="true" /></button>
            </div>
            <div className="table-wrap">
              <div className="table-scroll">
                <table className="dataset-table">
                  <thead><tr><th>数据集名称</th><th>任务类型</th><th>数据量</th><th>状态</th><th>更新时间</th><th>创建人</th><th>操作</th></tr></thead>
                  <tbody>{pagedDatasets.map((item) => (
                    <tr key={item.id}>
                      <td><button className="dataset-link" onClick={() => setSelected(item)}>{item.name}</button><div className="row-sub">{item.subtitle}</div><div className="dataset-id-row"><code>{item.id}</code><CopyDatasetId id={item.id} /></div></td>
                      <td>{item.type}{item.structure ? ` / ${item.structure}` : ""}</td>
                      <td>{item.count}</td>
                      <td><span className={`status ${stateMeta[item.state].className}`}><span className="status-dot" />{stateMeta[item.state].label}</span></td>
                      <td>{item.updatedAt}</td><td>{item.owner}</td>
                      <td><div className="actions">{actionsFor(item).map((action) => <button key={action} className={`link-btn ${action === "删除" ? "danger" : ""}`} onClick={() => void handleRowAction(item, action)}>{action}</button>)}</div></td>
                    </tr>
                  ))}</tbody>
                </table>
                {filtered.length === 0 && <div className="empty">没有匹配的数据集</div>}
              </div>
              <Pagination page={datasetSafePage} totalPages={datasetTotalPages} total={filtered.length} unit="条数据" onPageChange={setDatasetPage} />
            </div>
          </section>
        )}

        {module === "datasets" && view === "preview" && previewDataset && (
          <DatasetPreview
            dataset={previewDataset}
            onBack={() => {
              setPreviewDataset(null);
              setView("list");
              void refreshDatasets();
            }}
          />
        )}

        {module === "datasets" && view === "create" && (
          <section className="content">
            <div className="title-row"><div><BackButton label="返回数据集列表" disabled={processing || uploading} onClick={requestCreateBack} /><h1>新建数据集</h1><div className="subtitle">在一个页面完成基本信息和文件上传，提交后由系统自动初始化并校验</div></div></div>
            <div className="create-layout">
              <div className="panel">
                <h2>数据集信息与文件</h2>
                <div className="field">
                  <label className="field-label">数据集名称 <span className="required">*</span></label>
                  <input className="input" value={name} onChange={(event) => { setName(event.target.value); setFormError(""); }} placeholder="例如：客服偏好数据 v1" />
                </div>
                <div className="field">
                  <label className="field-label">任务类型 <span className="required">*</span></label>
                  <div className="type-grid">
                    <button className={`type-card ${datasetType === "文本生成" ? "selected" : ""}`} onClick={() => changeDatasetType("文本生成")}><span className="type-icon">文</span><span><span className="type-title">文本生成</span><span className="type-desc">支持 SFT 指令数据与 DPO 偏好数据</span></span></button>
                    <button className={`type-card ${datasetType === "文生图" ? "selected" : ""}`} onClick={() => changeDatasetType("文生图")}><span className="type-icon">图</span><span><span className="type-title">文生图</span><span className="type-desc">图片 + Caption，生成 metadata.jsonl</span></span></button>
                  </div>
                </div>
                {datasetType === "文本生成" && datasetPurpose === "training" ? (
                  <div className="field">
                    <label className="field-label">文本数据结构 <span className="required">*</span></label>
                    <div className="segmented"><button className={structure === "SFT" ? "selected" : ""} onClick={() => changeStructure("SFT")}>SFT 指令微调</button><button className={structure === "DPO" ? "selected" : ""} onClick={() => changeStructure("DPO")}>DPO 偏好数据</button></div>
                    <div className="hint"><strong>{structure === "DPO" ? "DPO 格式：messages + chosen / rejected" : "SFT 格式：messages 数组"}</strong>{structure === "DPO" ? "支持你提供的对话格式：messages 为上下文，chosen 与 rejected 为 assistant 消息对象；也兼容 prompt / chosen / rejected 字符串格式。" : "每行一个 JSON 对象；system 可选，随后 user 与 assistant 交替，并以 assistant 结束。支持 assistant 内容中的 <think> 标签。"}</div>
                  </div>
                ) : datasetPurpose === "evaluation" ? (
                  <div className="field"><label className="field-label">评测集 JSONL 格式</label><div className="hint"><strong>推荐：prompt + completion。</strong>也兼容 question/reference、messages（user + assistant）以及 DPO prompt/chosen/rejected。每条记录会保留来源文件和行号；可上传最多 10 个文件。</div></div>
                ) : (
                  <div className="field"><label className="field-label">文生图处理方式</label><div className="hint"><strong>图片 + 自动 Caption</strong>支持 JPG、PNG、WebP；单张不超过 20MB，单个数据集最多 200 张。</div></div>
                )}
                <div className="field">
                  <label className="field-label">上传文件 <span className="required">*</span></label>
                  <div className="dropzone" onDragOver={(event) => event.preventDefault()} onDrop={handleDrop}>
                    <div className="dropzone-title">拖拽文件到这里，或从本地选择</div>
                    <div className="dropzone-sub">{datasetType === "文生图" ? "仅接受 JPG / PNG / WebP，单张 ≤ 20MB，最多 200 张" : datasetPurpose === "evaluation" ? "仅接受 UTF-8 JSONL，单文件 ≤ 200MB，最多 10 个文件；支持 prompt + completion" : "仅接受 UTF-8 JSONL，单文件 ≤ 2GB"}</div>
                    <label className="secondary-btn file-pick">选择文件<input type="file" hidden multiple={datasetType === "文生图" || datasetPurpose === "evaluation"} accept={accept} onChange={(event) => handleFileInput(event.target.files)} /></label>
                  </div>
                  {fileError && <div className="field-error">{fileError}</div>}
                  {files.length > 0 && (
                    <div className="file-list">
                      {files.slice(0, 5).map((file) => <div className="file-item" key={file.name}><div><div className="file-name">{file.name}</div><div className="file-meta">{formatBytes(file.size)}</div></div><span className={`file-status ${uploadProgress === 100 ? "done" : ""}`}>{uploadProgress === 100 ? "上传完成" : `${uploadProgress}%`}</span><div className="progress"><span style={{ width: `${uploadProgress}%` }} /></div></div>)}
                      {files.length > 5 && <div className="more-files">另外 {files.length - 5} 个文件已加入上传队列</div>}
                    </div>
                  )}
                  {datasetType === "文本生成" && datasetPurpose !== "evaluation" && files.length > 0 && (
                    <div className={`validation-preview ${validationErrors.length ? "has-error" : "is-valid"}`}>
                      <strong>{validationErrors.length ? `预检查发现 ${validationErrors.length} 项格式错误` : `预检查通过，共 ${validationRows} 条样本`}</strong>
                      {validationErrors.slice(0, 3).map((error) => <span key={error}>{error}</span>)}
                      {validationWarnings.map((warning) => <span className="validation-warning" key={warning}>{warning}</span>)}
                    </div>
                  )}
                </div>
                {formError && <div className="form-error">{formError}</div>}
                {processing && <div className="processing-notice"><span className="processing-spinner" />{processingStage}<small>处理完成后将自动返回数据集列表</small></div>}
                <div className="form-footer"><div /><div className="form-actions"><button className="primary-btn" disabled={uploading || processing || !files.length} onClick={() => void createAndProcess()}>{uploading ? `上传中 ${uploadProgress}%` : processing ? "正在处理" : "创建并校验"}</button></div></div>
              </div>
              <aside className="panel guide">
                <h2>处理流程</h2><p>上传完成后，系统会在当前页面短暂初始化并校验。只有处理完成的数据集才进入列表。</p>
                <div className="pipeline"><span>上传文件</span><b>→</b><span>初始化与校验</span><b>→</b><span>进入列表</span></div>
                <div className="guide-list">
                  <div className="guide-item"><span className="guide-num">1</span><span><strong>初始化</strong><span>解压文件、识别编码、统计样本并生成清单。</span></span></div>
                  <div className="guide-item"><span className="guide-num">2</span><span><strong>内容校验</strong><span>检查数据结构、必填字段、文件映射和格式限制。</span></span></div>
                  <div className="guide-item"><span className="guide-num">3</span><span><strong>发布</strong><span>校验通过后由有权限的成员手动发布。</span></span></div>
                </div>
                <div className="warning"><strong>测试校验失败状态</strong>上传文件名包含 invalid 或 error 的文件，最终会进入“校验失败”。</div>
              </aside>
            </div>
          </section>
        )}
        {module === "training" && <TrainingModule datasets={datasets} datasetsLoaded={datasetsLoaded} datasetsLoading={datasetsLoading} datasetsError={datasetsError} onLoadDatasets={refreshDatasets} onOpenDatasets={() => { setModule("datasets"); setView("list"); }} createRequest={trainingCreateRequest} onCreateRequestHandled={(requestId) => setTrainingCreateRequest((current) => current?.requestId === requestId ? null : current)} />}
        {module === "evaluation" && evaluationEnabled && <EvaluationModule />}
        {module === "models" && <ModelManagementModule view={modelView} onOpenTraining={openTrainingWithModel} onOpenInference={openInferenceWithArtifact} />}
        {module === "inference" && <InferenceModule createRequest={inferenceCreateRequest} onCreateRequestHandled={(requestId) => setInferenceCreateRequest((current) => current?.requestId === requestId ? null : current)} />}
      </main>

      {selected && module === "datasets" && view === "list" && (
        <>
        <button type="button" className="detail-drawer-backdrop" aria-label="关闭数据集详情" onClick={() => setSelected(null)} />
        <div ref={datasetDrawerRef} className="detail-drawer" role="dialog" aria-modal="true" aria-labelledby="dataset-drawer-title" tabIndex={-1}>
          <div className="drawer-head"><div><h2 id="dataset-drawer-title">{selected.name}</h2><div className="subtitle">{selected.type}{selected.structure ? ` / ${selected.structure}` : ""}</div></div><button type="button" className="close-btn" aria-label="关闭数据集详情" title="关闭数据集详情" onClick={() => setSelected(null)}><X size={18} aria-hidden="true" /></button></div>
          <div className="detail-grid"><div><div className="detail-label">状态</div><div className="detail-value"><span className={`status ${stateMeta[selected.state].className}`}>{stateMeta[selected.state].label}</span></div></div><div><div className="detail-label">数据集 ID</div><div className="detail-value dataset-id-value"><code>{selected.id}</code><CopyDatasetId id={selected.id} /></div></div><div><div className="detail-label">数据量</div><div className="detail-value">{selected.count}</div></div><div><div className="detail-label">创建人</div><div className="detail-value">{selected.owner}</div></div><div><div className="detail-label">更新时间</div><div className="detail-value">{selected.updatedAt}</div></div></div>
          <div className="drawer-section"><h3>校验结果</h3><p className="subtitle">{selected.validationMessage ?? (selected.state === "failed" ? "格式错误，请修改后重新上传。" : "校验通过，当前版本可用于微调任务。")}</p>{selected.warningCount ? <p className="validation-warning">另有 {selected.warningCount} 类警告，不阻止发布。</p> : null}</div>
          {selected.state === "failed" && <div className="drawer-section"><h3>修复数据集</h3><DatasetRepairPanel dataset={selected} onUpdated={(updated) => { setDatasets((items) => items.map((item) => item.id === selected.id ? updated : item)); setSelected(updated); }} /></div>}
          <div className="drawer-section"><h3>操作</h3><div className="form-actions">{selected.state === "ready" && <button className="primary-btn" onClick={publish}>发布</button>}{selected.state !== "published" && <button className="danger-btn" onClick={() => setModal("delete")}>删除</button>}</div></div>
        </div>
        </>
      )}

      {modal === "publish" && <div className="modal-backdrop"><div ref={datasetModalRef} className="modal" role="dialog" aria-modal="true" aria-labelledby="publish-title" tabIndex={-1}><h2 id="publish-title">发布数据集</h2><p>发布后数据集文件和校验结论将被冻结，后续修改请重新上传或在校验失败时修复。确认发布“{selected?.name}”吗？</p><div className="modal-actions"><button type="button" className="secondary-btn" onClick={() => setModal(null)}>取消</button><button type="button" className="primary-btn" onClick={() => void confirmPublish()}>确认发布</button></div></div></div>}
      {modal === "success" && <div className="modal-backdrop"><div ref={datasetModalRef} className="modal" role="dialog" aria-modal="true" aria-labelledby="success-title" tabIndex={-1}><h2 id="success-title">发布成功</h2><p>数据集已发布，现在可以在微调任务中选择它。</p><div className="modal-actions"><button type="button" className="primary-btn" onClick={() => { setModal(null); setSelected(null); }}>完成</button></div></div></div>}
      {modal === "delete" && <div className="modal-backdrop"><div ref={datasetModalRef} className="modal" role="dialog" aria-modal="true" aria-labelledby="delete-title" tabIndex={-1}><h2 id="delete-title">确认删除</h2><p>将删除“{selected?.name}”。如果该数据集已被任务引用，正式环境会改为归档。</p><div className="modal-actions"><button type="button" className="secondary-btn" onClick={() => setModal(null)}>取消</button><button type="button" className="danger-btn" onClick={() => void deleteDataset()}>确认删除</button></div></div></div>}
      {modal === "import" && <div className="modal-backdrop"><div ref={datasetModalRef} className="modal import-modal" role="dialog" aria-modal="true" aria-labelledby="import-title" tabIndex={-1}><div className="modal-head"><div><h2 id="import-title">导入数据集</h2><p>从 MinIO rag-assets 桶的 exports/ 目录选择 JSONL 文件，导入后会沿用现有校验和发布流程。</p></div><button type="button" className="close-btn" aria-label="关闭导入窗口" title="关闭导入窗口" onClick={() => { if (!importSubmitting) setModal(null); }} disabled={importSubmitting}><X size={18} aria-hidden="true" /></button></div>{importLoading ? <div className="empty">正在读取 exports/ 目录...</div> : <><div className="field"><label className="field-label">选择文件 <span className="required">*</span></label><select className="input" value={importSourceKey} onChange={(event) => selectImportSource(event.target.value)} disabled={importSubmitting}><option value="">请选择 JSONL 文件</option>{importSources.map((source) => <option key={source.key} value={source.key} disabled={!source.supported}>{source.key} · {formatImportTime(source.lastModified)}{source.supported ? "" : "（暂不支持）"}</option>)}</select></div><div className="field"><label className="field-label">数据集名称 <span className="required">*</span></label><input className="input" value={importName} onChange={(event) => setImportName(event.target.value)} placeholder="例如：客服偏好数据 v1" disabled={importSubmitting} /></div><div className="field"><label className="field-label">文本数据结构 <span className="required">*</span></label><div className="segmented"><button type="button" className={importStructure === "SFT" ? "selected" : ""} onClick={() => setImportStructure("SFT")} disabled={importSubmitting}>SFT 指令微调</button><button type="button" className={importStructure === "DPO" ? "selected" : ""} onClick={() => setImportStructure("DPO")} disabled={importSubmitting}>DPO 偏好数据</button></div></div>{importError && <div className="form-error">{importError}</div>}<div className="modal-actions"><button type="button" className="secondary-btn" onClick={() => setModal(null)} disabled={importSubmitting}>取消</button><button type="button" className="primary-btn" onClick={() => void importDataset()} disabled={importSubmitting || importLoading || !importSourceKey}>{importSubmitting ? "正在导入" : "导入并校验"}</button></div></>}</div></div>}
      {modal === "leave-create" && <div className="modal-backdrop"><div ref={datasetModalRef} className="modal" role="dialog" aria-modal="true" aria-labelledby="leave-title" tabIndex={-1}><h2 id="leave-title">确认返回</h2><p>当前填写内容尚未提交，返回数据集列表后将丢弃这些内容。</p><div className="modal-actions"><button type="button" className="secondary-btn" onClick={() => setModal(null)}>继续编辑</button><button type="button" className="primary-btn" onClick={leaveCreate}>继续返回</button></div></div></div>}
    </div>
  );
}
