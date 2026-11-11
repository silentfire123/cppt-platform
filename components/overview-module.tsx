"use client";

import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { apiUrl } from "../lib/api-config";
import { formatApiError } from "../lib/api-error";

type TargetModule = "datasets" | "training" | "models" | "inference";

type OverviewSnapshot = {
  status: "healthy" | "attention";
  generatedAt: string;
  counts: {
    datasets: { total: number; published: number; ready: number; failed: number };
    training: { total: number; active: number; queued: number; completed: number; failed: number; stopped: number };
    models: { total: number; base: number; trained: number; unavailable: number };
    inference: { total: number; running: number; online: number; failed: number };
  };
  runtime: {
    worker: string;
    activeTaskId?: string | null;
    activeTaskIds?: string[];
    queue: string;
    storageBackend: string;
    inference: string;
    resourceOwner: "idle" | "training" | "inference";
    sharedStorage: { status: "ready" | "unavailable"; error?: string | null };
    gpu: {
      available: boolean;
      name?: string;
      reason?: string;
      memoryTotalMb?: number;
      memoryUsedMb?: number;
      memoryFreeMb?: number;
      memoryUsagePercent?: number | null;
    };
    youyun?: {
      activeCount: number;
      trainingCount: number;
      inferenceCount: number;
      gpuTypes: string[];
      postpaidHourlyTotal: number;
      prepaidCount: number;
      currency: string;
    };
  };
  activeTraining?: {
    id: string;
    name: string;
    state: string;
    progress: number;
    currentStep: string;
    model: string;
    provider: string;
    executionMode: string;
    gpuType?: string | null;
    createdAt?: string;
    error?: string | null;
  } | null;
  activeInference?: {
    id: string;
    name: string;
    state: string;
    exposure: string;
    model: string;
    provider?: string;
    adapterVariant: string;
    gpu: string;
    memory: string;
    updatedAt?: string;
    error?: string | null;
  } | null;
  inferenceMetrics?: {
    runningRequests?: number | null;
    waitingRequests?: number | null;
    batchSize?: number | null;
    ttftMs?: number | null;
    tpotMs?: number | null;
    decodeTokensPerSecond?: number | null;
    kvCacheUsagePercent?: number | null;
    prefixCacheHitRate?: number | null;
  } | null;
  alerts: Array<{
    id: string;
    severity: "critical" | "warning";
    module: TargetModule | "overview";
    entityId?: string;
    title: string;
    detail: string;
  }>;
  recentActivity: Array<{
    id: string;
    type: "dataset" | "training" | "model" | "inference";
    module: TargetModule;
    name: string;
    state: string;
    timestamp?: string;
  }>;
};

const moduleLabel: Record<TargetModule, string> = {
  datasets: "数据集",
  training: "微调任务",
  models: "模型管理",
  inference: "推理服务",
};

const stateLabels: Record<string, string> = {
  published: "已发布",
  ready: "待发布",
  failed: "失败",
  queued: "排队中",
  provisioning: "资源准备中",
  preparing: "环境准备中",
  running: "运行中",
  stopping: "停止中",
  collecting: "收集产物",
  uploading_artifact: "上传产物",
  completed: "已完成",
  stopped: "已停止",
  available: "可用",
  unavailable: "不可用",
  created: "待启动",
  recovering: "恢复中",
  online: "已上线",
};

function formatTime(value?: string) {
  if (!value) return "--";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString("zh-CN", { hour12: false });
}

function formatMemory(value?: number) {
  if (typeof value !== "number") return "--";
  return `${(value / 1024).toFixed(value >= 10240 ? 0 : 1)} GB`;
}

function formatNumber(value?: number | null, suffix = "") {
  if (typeof value !== "number" || !Number.isFinite(value)) return "--";
  return `${value.toFixed(value >= 100 ? 0 : 1)}${suffix}`;
}

function formatPercent(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "--";
  const percent = value <= 1 ? value * 100 : value;
  return `${percent.toFixed(1)}%`;
}

function decodeSpeed(metrics?: OverviewSnapshot["inferenceMetrics"]) {
  if (typeof metrics?.decodeTokensPerSecond === "number") return metrics.decodeTokensPerSecond;
  if (typeof metrics?.tpotMs === "number" && metrics.tpotMs > 0) return 1000 / metrics.tpotMs;
  return null;
}

export function OverviewModule({ onOpenModule }: { onOpenModule: (module: TargetModule) => void }) {
  const [snapshot, setSnapshot] = useState<OverviewSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async (initial = false) => {
    if (initial) setLoading(true);
    else setRefreshing(true);
    try {
      const response = await fetch(apiUrl("/overview"), { credentials: "include", cache: "no-store" });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(formatApiError(body, `概览接口请求失败（${response.status}）`));
      }
      setSnapshot((await response.json()) as OverviewSnapshot);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "系统概览暂不可用。");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    let timer: number | null = null;
    const startPolling = () => {
      if (timer !== null) window.clearInterval(timer);
      timer = null;
      if (document.visibilityState === "visible") {
        timer = window.setInterval(() => void refresh(), 10_000);
      }
    };
    const handleVisibility = () => {
      if (document.visibilityState === "visible") void refresh();
      startPolling();
    };
    void refresh(true);
    startPolling();
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      if (timer !== null) window.clearInterval(timer);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [refresh]);

  const status = error && !snapshot ? "unavailable" : snapshot?.status ?? "loading";
  const statusCopy = status === "healthy"
    ? { label: "运行正常", detail: "核心服务与业务数据已连接" }
    : status === "attention"
      ? { label: "存在待处理项", detail: "请检查下方异常与资源状态" }
      : status === "unavailable"
        ? { label: "概览暂不可用", detail: error }
        : { label: "正在读取", detail: "正在汇总平台实时状态" };
  const gpu = snapshot?.runtime.gpu;
  const youyun = snapshot?.runtime.youyun;
  const youyunActive = (youyun?.activeCount ?? 0) > 0;
  const youyunGpuTypes = youyun?.gpuTypes?.length ? youyun.gpuTypes.join(" / ") : "";
  const youyunCurrency = youyun?.currency === "CNY" || !youyun?.currency ? "¥" : `${youyun.currency} `;
  const youyunBilling = !youyun || !youyunActive
    ? "--"
    : youyun.postpaidHourlyTotal > 0
      ? `${youyunCurrency}${youyun.postpaidHourlyTotal.toFixed(2)}/时`
      : youyun.prepaidCount > 0
        ? `${youyun.prepaidCount} 个包月`
        : "待查询";
  const youyunChargeType = !youyun || !youyunActive
    ? "--"
    : youyun.postpaidHourlyTotal > 0 && youyun.prepaidCount > 0
      ? "按量 + 包月"
      : youyun.prepaidCount > 0
        ? "包月"
        : "按量计费";
  const localGpuNote = gpu?.available
    ? `本机 GPU：${gpu.name} · 已用 ${formatMemory(gpu.memoryUsedMb)} / 总计 ${formatMemory(gpu.memoryTotalMb)}`
    : "";
  const billingNote = youyunActive ? "优云费率为运行中实例的按量单价合计，实际以优云账单为准" : "";
  const metrics = snapshot?.inferenceMetrics;

  const cards = [
    {
      key: "datasets" as const,
      title: "数据集",
      total: snapshot?.counts.datasets.total,
      items: [
        ["已发布", snapshot?.counts.datasets.published],
        ["待发布", snapshot?.counts.datasets.ready],
        ["校验失败", snapshot?.counts.datasets.failed],
      ],
    },
    {
      key: "training" as const,
      title: "微调任务",
      total: snapshot?.counts.training.total,
      items: [
        ["运行 / 排队", snapshot ? snapshot.counts.training.active + snapshot.counts.training.queued : undefined],
        ["已完成", snapshot?.counts.training.completed],
        ["失败", snapshot?.counts.training.failed],
      ],
    },
    {
      key: "models" as const,
      title: "模型资产",
      total: snapshot?.counts.models.total,
      items: [
        ["基础模型", snapshot?.counts.models.base],
        ["微调模型", snapshot?.counts.models.trained],
        ["不可用", snapshot?.counts.models.unavailable],
      ],
    },
    {
      key: "inference" as const,
      title: "推理服务",
      total: snapshot?.counts.inference.total,
      items: [
        ["运行中", snapshot?.counts.inference.running],
        ["已上线", snapshot?.counts.inference.online],
        ["启动失败", snapshot?.counts.inference.failed],
      ],
    },
  ];

  return (
    <section className="content overview-page">
      <div className="title-row overview-title-row">
        <div><h1>系统概览</h1><div className="subtitle">监控数据资产、训练、模型、推理服务与当前计算资源</div></div>
        <div className="overview-refresh-group"><span>每 10 秒自动刷新</span><button type="button" className="secondary-btn refresh" title="刷新系统概览" aria-label="刷新系统概览" disabled={refreshing} onClick={() => void refresh()}>{refreshing ? "…" : <RefreshCw size={15} aria-hidden="true" />}</button></div>
      </div>

      <div className={`overview-health overview-health-${status}`}>
        <span className="overview-health-dot" />
        <div><strong>{statusCopy.label}</strong><span>{statusCopy.detail}</span></div>
        <div className="overview-health-meta"><span>资源占用：{snapshot?.runtime.resourceOwner === "training" ? "训练任务" : snapshot?.runtime.resourceOwner === "inference" ? "推理服务" : snapshot ? "空闲" : "--"}</span><span>更新于 {formatTime(snapshot?.generatedAt)}</span></div>
      </div>
      {error && snapshot && <div className="overview-stale-notice">本次刷新失败，当前展示上一次成功快照：{error}</div>}

      <div className="overview-domain-grid">
        {cards.map((card) => <button key={card.key} className="overview-domain-card" onClick={() => onOpenModule(card.key)}><div className="overview-domain-head"><span>{card.title}</span><b>查看</b></div><strong className="overview-domain-total">{card.total ?? "--"}</strong><div className="overview-domain-items">{card.items.map(([label, value]) => <span key={label}><small>{label}</small><b>{value ?? "--"}</b></span>)}</div></button>)}
      </div>

      <div className="overview-two-column">
        <section className="panel overview-runtime-panel">
          <div className="overview-section-head"><div><h2>计算资源</h2><p>优云 GPU 实例与按量计费状态</p></div><span className="overview-owner">{youyunActive ? "实例运行中" : "优云空闲"}</span></div>
          <div className="overview-gpu-row">
            <div className={`overview-gpu-mark ${youyunActive ? "available" : "unavailable"}`}>优云</div>
            <div className="overview-gpu-copy"><strong>{youyunActive ? `优云实例运行中${youyunGpuTypes ? ` · ${youyunGpuTypes}` : ""}` : "当前没有运行中的优云实例"}</strong><span>{youyunActive ? `训练实例 ${youyun?.trainingCount ?? 0} 个 · 推理实例 ${youyun?.inferenceCount ?? 0} 个` : "优云按量计费，空闲期间不产生费用"}</span></div>
            <b>{youyunBilling}</b>
          </div>
          <div className="overview-runtime-grid">
            <div><span>训练实例</span><strong>{youyun ? youyun.trainingCount : "--"}</strong></div>
            <div><span>推理实例</span><strong>{youyun ? youyun.inferenceCount : "--"}</strong></div>
            <div><span>占用型号</span><strong>{youyunGpuTypes || "--"}</strong></div>
            <div><span>计费方式</span><strong>{youyunChargeType}</strong></div>
          </div>
          <p className="overview-boundary-note">{[localGpuNote, billingNote].filter(Boolean).join("；") || "优云实例按量计费，实际费用以优云账单为准。"}</p>
        </section>

        <section className="panel overview-active-panel">
          <div className="overview-section-head"><div><h2>当前负载</h2><p>活动训练和推理服务</p></div></div>
          <div className="overview-active-list">
            <button onClick={() => onOpenModule("training")}><span className="overview-active-type">训练</span>{snapshot?.activeTraining ? <div><strong>{snapshot.activeTraining.name}</strong><small>{stateLabels[snapshot.activeTraining.state] ?? snapshot.activeTraining.state} · {snapshot.activeTraining.progress}% · {snapshot.activeTraining.currentStep}</small><em>{snapshot.activeTraining.model}{snapshot.activeTraining.gpuType ? ` · ${snapshot.activeTraining.gpuType}` : ""}</em></div> : <div><strong>当前没有活动训练</strong><small>本地训练资源处于空闲状态</small></div>}<b>›</b></button>
            <button onClick={() => onOpenModule("inference")}><span className="overview-active-type inference">推理</span>{snapshot?.activeInference ? <div><strong>{snapshot.activeInference.name}</strong><small>{stateLabels[snapshot.activeInference.state] ?? snapshot.activeInference.state} · {snapshot.activeInference.gpu || "资源分配中"} · {snapshot.activeInference.memory || "显存读取中"}</small><em>{snapshot.activeInference.model} · {snapshot.activeInference.adapterVariant === "final" ? "Final" : "Best"} Adapter</em></div> : <div><strong>当前没有活动推理</strong><small>没有启动中的 vLLM 服务</small></div>}<b>›</b></button>
          </div>
        </section>
      </div>

      <section className="panel overview-performance-panel">
        <div className="overview-section-head"><div><h2>推理实时指标</h2><p>{snapshot?.activeInference?.state === "running" ? `来自 ${snapshot.activeInference.name} 的 vLLM 指标` : "推理服务运行后展示实时性能"}</p></div><span>{snapshot?.activeInference?.state === "running" ? "实时" : "未运行"}</span></div>
        <div className="overview-performance-grid">
          <div><span>运行请求</span><strong>{formatNumber(metrics?.runningRequests)}</strong><small>正在生成</small></div>
          <div><span>排队请求</span><strong>{formatNumber(metrics?.waitingRequests)}</strong><small>等待调度</small></div>
          <div><span>Batch Size</span><strong>{formatNumber(metrics?.batchSize)}</strong><small>当前批大小</small></div>
          <div><span>TTFT</span><strong>{formatNumber(metrics?.ttftMs, " ms")}</strong><small>首 Token 延迟</small></div>
          <div><span>Decode</span><strong>{formatNumber(decodeSpeed(metrics), " tok/s")}</strong><small>生成吞吐</small></div>
          <div><span>KV Cache</span><strong>{formatPercent(metrics?.kvCacheUsagePercent)}</strong><small>缓存使用率</small></div>
          <div><span>Prefix Cache</span><strong>{formatPercent(metrics?.prefixCacheHitRate)}</strong><small>前缀命中率</small></div>
        </div>
      </section>

      <div className="overview-two-column overview-bottom-grid">
        <section className="panel overview-list-panel">
          <div className="overview-section-head"><div><h2>异常与待处理</h2><p>只提供定位，不在概览执行破坏性操作</p></div><span>{snapshot?.alerts.length ?? "--"} 项</span></div>
          <div className="overview-alert-list">{snapshot?.alerts.length ? snapshot.alerts.map((alert) => <button key={alert.id} onClick={() => { if (alert.module !== "overview") onOpenModule(alert.module); }}><i className={alert.severity} /><div><strong>{alert.title}</strong><span>{alert.detail}</span></div>{alert.module !== "overview" && <b>{moduleLabel[alert.module]} ›</b>}</button>) : <div className="overview-empty-state">当前没有待处理异常</div>}</div>
        </section>
        <section className="panel overview-list-panel">
          <div className="overview-section-head"><div><h2>最近动态</h2><p>来自现有任务和资产时间字段</p></div></div>
          <div className="overview-activity-list">{snapshot?.recentActivity.length ? snapshot.recentActivity.map((item) => <button key={`${item.type}-${item.id}`} onClick={() => onOpenModule(item.module)}><span>{item.type === "training" ? "训" : item.type === "inference" ? "推" : item.type === "model" ? "模" : "数"}</span><div><strong>{item.name}</strong><small>{moduleLabel[item.module]} · {(stateLabels[item.state] ?? item.state) || "已更新"}</small></div><time>{formatTime(item.timestamp)}</time></button>) : <div className="overview-empty-state">暂无最近动态</div>}</div>
        </section>
      </div>
    </section>
  );
}
