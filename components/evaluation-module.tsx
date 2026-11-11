"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { CheckCircle2, ClipboardCheck, Play, RefreshCw, XCircle } from "lucide-react";
import { apiUrl } from "../lib/api-config";
import { formatApiError } from "../lib/api-error";
import type { Dataset, TrainingTask } from "../lib/types";

type EvaluationState = "draft" | "queued" | "generating" | "waiting_human_review" | "completed" | "failed" | "cancelled";
type EvaluationTask = { id: string; name: string; datasetId: string; datasetVersion: number; targetTaskId: string; targetVariant: "selected" | "best" | "final"; state: EvaluationState; totalItems: number; generatedItems: number; reviewedItems: number; error?: string | null; judgeConfig: { baseUrl?: string; model?: string; useTeacher: boolean; useSimilarity: boolean }; dimensions: string[]; createdAt: string; };
type EvaluationItem = { id: string; sourceIndex: number; sourceFile?: string | null; sourceLine?: number | null; prompt: string; referenceAnswer?: string | null; modelAnswer?: string | null; teacherResult?: { score?: number; dimensions?: Record<string, number>; reason?: string; error?: string; status?: string }; similarityResult?: { score?: number; kind?: string; status?: string }; humanResult?: { score?: number; dimensions?: Record<string, number> }; attribution?: string | null; reviewNote?: string | null; status: string; };
type Summary = { total: number; reviewed: number; coverage: number; human: { mean: number | null; variance: number | null; count: number }; teacher: { mean: number | null; variance: number | null; count: number }; similarity: { mean: number | null; variance: number | null; count: number }; attribution: Record<string, number>; lowScoreCount: number; lowSimilarityCount: number };

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), { ...options, credentials: "include", cache: "no-store" });
  if (!response.ok) throw new Error(formatApiError(await response.json().catch(() => ({})), `请求失败（${response.status}）`));
  return response.json() as Promise<T>;
}

const attributionOptions = [
  ["model_wrong", "模型回答错误"], ["reference_wrong", "标准答案错误"], ["acceptable_different", "答案合理但表达不同"], ["both_wrong", "两者都不可靠"], ["needs_expert", "需要专家判断"], ["other", "其他"],
] as const;

function value(value: number | null | undefined, digits = 2) { return typeof value === "number" ? value.toFixed(digits) : "—"; }

export function EvaluationModule() {
  const [tasks, setTasks] = useState<EvaluationTask[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [trainingTasks, setTrainingTasks] = useState<TrainingTask[]>([]);
  const [selected, setSelected] = useState<EvaluationTask | null>(null);
  const [items, setItems] = useState<EvaluationItem[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [targetTaskId, setTargetTaskId] = useState("");
  const [variant, setVariant] = useState<"selected" | "best" | "final">("selected");
  const [judgeBaseUrl, setJudgeBaseUrl] = useState("");
  const [judgeModel, setJudgeModel] = useState("");
  const [useTeacher, setUseTeacher] = useState(true);
  const [useSimilarity, setUseSimilarity] = useState(true);
  const [creating, setCreating] = useState(false);

  const evaluationDatasets = useMemo(() => datasets.filter((dataset) => dataset.purpose === "evaluation" && dataset.state === "published"), [datasets]);
  const completedTraining = useMemo(() => trainingTasks.filter((task) => task.state === "completed" && task.provider === "youyun" && task.artifacts?.some(artifact => artifact.type === "adapter" && Boolean(artifact.path || artifact.remotePath)) && task.artifacts?.some(artifact => artifact.type === "youyun-instance" && Boolean(artifact.instanceId) && artifact.state === "awaiting_comparison")), [trainingTasks]);

  const load = useCallback(async () => {
    setLoading(true); setError("");
    try {
      const [taskRows, datasetRows, trainingResponse] = await Promise.all([request<EvaluationTask[]>("/evaluations"), request<Dataset[]>("/datasets"), request<TrainingTask[]>("/evaluations/targets")]);
      setTasks(taskRows); setDatasets(datasetRows); setTrainingTasks(trainingResponse);
      setSelected((current) => current ? taskRows.find((task) => task.id === current.id) ?? null : null);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "读取评测数据失败。"); }
    finally { setLoading(false); }
  }, []);

  const loadDetail = useCallback(async (task: EvaluationTask) => {
    setSelected(task); setError("");
    try {
      const [itemRows, nextSummary] = await Promise.all([request<EvaluationItem[]>(`/evaluations/${task.id}/items?limit=500`), request<Summary>(`/evaluations/${task.id}/summary`)]);
      setItems(itemRows); setSummary(nextSummary);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "读取评测详情失败。"); }
  }, []);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    if (!selected || !["queued", "generating"].includes(selected.state)) return;
    const timer = window.setInterval(() => {
      void request<EvaluationTask>(`/evaluations/${selected.id}`).then((task) => {
        setSelected(task);
        setTasks((current) => current.map((row) => row.id === task.id ? task : row));
        void loadDetail(task);
      }).catch((reason) => setError(reason instanceof Error ? reason.message : "刷新评测状态失败。"));
    }, 3000);
    return () => window.clearInterval(timer);
  }, [loadDetail, selected]);

  const create = async () => {
    if (!name.trim() || !datasetId || !targetTaskId) { setError("请填写任务名称，并选择已发布评测集和已完成训练任务。"); return; }
    setCreating(true); setError("");
    try {
      const task = await request<EvaluationTask>("/evaluations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: name.trim(), datasetId, targetTaskId, targetVariant: variant, dimensions: ["correctness", "relevance", "completeness"], judgeBaseUrl: judgeBaseUrl.trim() || null, judgeModel: judgeModel.trim() || null, useTeacher, useSimilarity }) });
      setTasks((current) => [task, ...current]); setName(""); await loadDetail(task);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "创建评测任务失败。"); }
    finally { setCreating(false); }
  };

  const start = async () => {
    if (!selected) return;
    try { const task = await request<EvaluationTask>(`/evaluations/${selected.id}/start`, { method: "POST" }); setSelected(task); setTasks((current) => current.map((row) => row.id === task.id ? task : row)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "启动评测失败。"); }
  };

  const review = async (item: EvaluationItem, score: number, attribution: string, note: string) => {
    if (!selected) return;
    try {
      const updated = await request<EvaluationItem>(`/evaluations/${selected.id}/items/${item.id}/review`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ score, dimensions: {}, attribution, note }) });
      setItems((current) => current.map((row) => row.id === updated.id ? updated : row));
      setSummary(await request<Summary>(`/evaluations/${selected.id}/summary`));
      await load();
    } catch (reason) { setError(reason instanceof Error ? reason.message : "保存人工审核失败。"); }
  };

  if (selected) return <EvaluationDetail task={selected} items={items} summary={summary} error={error} onBack={() => { setSelected(null); setItems([]); setSummary(null); }} canStart={completedTraining.some(task => task.id === selected.targetTaskId)} onStart={() => void start()} onReview={review} />;

  return <section className="content tuning-content">
    <div className="detail-title-row"><div><h1>模型评测</h1><div className="subtitle">固定评测集版本，分别记录 Teacher、语义相似度和人工最终结论。</div></div><button className="secondary-btn" onClick={() => void load()} disabled={loading}><RefreshCw size={16} />刷新</button></div>
    {error && <div className="form-error">{error}</div>}
    <section className="panel"><div className="section-heading"><div><h2>创建评测任务</h2><p>仅选择已完成且保留优云实例和 Adapter 的训练任务；API Key 由服务端环境变量保存。</p></div><span className="section-index">01</span></div>
      <div className="form-grid"><div className="field"><label className="field-label">任务名称 *</label><input className="input" value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：客服模型 v2 回归评测" /></div><div className="field"><label className="field-label">评测集 *</label><select className="input" value={datasetId} onChange={(event) => setDatasetId(event.target.value)}><option value="">请选择已发布评测集</option>{evaluationDatasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name} · v{dataset.version ?? 1} · {dataset.count}</option>)}</select></div><div className="field"><label className="field-label">被测训练任务 *</label><select className="input" value={targetTaskId} onChange={(event) => setTargetTaskId(event.target.value)}><option value="">请选择已完成训练任务</option>{completedTraining.map((task) => <option key={task.id} value={task.id}>{task.name} · {task.model}</option>)}</select></div><div className="field"><label className="field-label">Adapter 版本</label><select className="input" value={variant} onChange={(event) => setVariant(event.target.value as typeof variant)}><option value="selected">当前选中</option><option value="best">最佳</option><option value="final">最终</option></select></div></div>
      <div className="form-grid"><div className="field"><label className="field-label">Teacher Base URL（可选）</label><input className="input" value={judgeBaseUrl} onChange={(event) => setJudgeBaseUrl(event.target.value)} placeholder="服务端已配置时可留空" /></div><div className="field"><label className="field-label">Teacher 模型（可选）</label><input className="input" value={judgeModel} onChange={(event) => setJudgeModel(event.target.value)} placeholder="例如 qwen-plus" /></div></div>
      <div className="segmented"><button type="button" className={useTeacher ? "selected" : ""} onClick={() => setUseTeacher((current) => !current)}>综合评测 {useTeacher ? "已启用" : "未启用"}</button><button type="button" className={useSimilarity ? "selected" : ""} onClick={() => setUseSimilarity((current) => !current)}>语义相似度 {useSimilarity ? "已启用" : "未启用"}</button><span className="subtitle">人工核验始终启用，最终平均分与方差仅基于人工确认。</span></div>
      <div className="form-actions"><button className="primary-btn" onClick={() => void create()} disabled={creating}>{creating ? "创建中" : "创建评测任务"}</button></div>
    </section>
    <section className="panel"><div className="section-heading"><div><h2>评测任务</h2><p>自动评分仅用于预标；需逐条人工确认后才完成。</p></div><span className="section-index">02</span></div>{loading ? <div className="empty">加载中...</div> : tasks.length ? <div className="task-list">{tasks.map((task) => <button type="button" className="task-card" key={task.id} onClick={() => void loadDetail(task)}><span className="model-avatar trained"><ClipboardCheck size={17} /></span><span><strong>{task.name}</strong><small>{task.state} · 已人工确认 {task.reviewedItems}/{task.totalItems} · 已生成 {task.generatedItems}/{task.totalItems}</small></span><span className="status status-ready">{task.state}</span></button>)}</div> : <div className="empty">暂无评测任务。请先上传并发布“评测集 / 测试集”。</div>}</section>
  </section>;
}

function EvaluationDetail({ task, items, summary, error, onBack, canStart, onStart, onReview }: { task: EvaluationTask; items: EvaluationItem[]; summary: Summary | null; error: string; onBack: () => void; canStart: boolean; onStart: () => void; onReview: (item: EvaluationItem, score: number, attribution: string, note: string) => Promise<void>; }) {
  return <section className="content tuning-content"><div className="detail-title-row"><div><button className="back-link" onClick={onBack}>返回评测任务</button><h1>{task.name}</h1><div className="subtitle">状态：{task.state} · 人工确认 {task.reviewedItems}/{task.totalItems}</div></div>{task.state === "draft" && <button className="primary-btn" disabled={!canStart} title={canStart ? undefined : "优云训练实例已释放或 Adapter 不可用"} onClick={onStart}><Play size={16} />开始评测</button>}</div>{error && <div className="form-error">{error}</div>}
    <div className="overview-grid">{[["人工平均分", value(summary?.human.mean)], ["人工方差", value(summary?.human.variance)], ["审核覆盖率", `${((summary?.coverage ?? 0) * 100).toFixed(0)}%`], ["Teacher 平均分", value(summary?.teacher.mean)], ["相似度均值", value(summary?.similarity.mean)], ["低相似度", String(summary?.lowSimilarityCount ?? 0)]].map(([label, result]) => <div className="panel stat-card" key={label}><span>{label}</span><strong>{result}</strong></div>)}</div>
    <section className="panel"><div className="section-heading"><div><h2>逐条人工核验</h2><p>模型回答合理而标准答案错误时，请选择对应归因；人工分数才进入最终统计。</p></div><span className="section-index">03</span></div>{items.map((item) => <ReviewRow key={item.id} item={item} onReview={onReview} />)}{!items.length && <div className="empty">任务开始后将在这里显示逐条结果。</div>}</section>
  </section>;
}

function ReviewRow({ item, onReview }: { item: EvaluationItem; onReview: (item: EvaluationItem, score: number, attribution: string, note: string) => Promise<void>; }) {
  const [score, setScore] = useState(item.humanResult?.score ?? item.teacherResult?.score ?? 0);
  const [attribution, setAttribution] = useState(item.attribution ?? "model_wrong");
  const [note, setNote] = useState(item.reviewNote ?? "");
  const [saving, setSaving] = useState(false);
  return <article className="drawer-section" style={{ borderTop: "1px solid var(--line, #e7edf8)", paddingTop: 20 }}><div className="detail-title-row"><strong>#{item.sourceIndex} {item.sourceFile ? `· ${item.sourceFile}:${item.sourceLine}` : ""}</strong><span className={`status ${item.status === "reviewed" ? "status-published" : "status-ready"}`}>{item.status === "reviewed" ? "已人工确认" : item.status}</span></div><div className="detail-grid"><div><div className="detail-label">问题</div><div className="detail-value">{item.prompt}</div></div><div><div className="detail-label">标准答案</div><div className="detail-value">{item.referenceAnswer || "—"}</div></div><div><div className="detail-label">模型回答</div><div className="detail-value">{item.modelAnswer || "等待生成"}</div></div></div><div className="subtitle">Teacher：{value(item.teacherResult?.score)} {item.teacherResult?.reason ? `· ${item.teacherResult.reason}` : item.teacherResult?.error ? `· ${item.teacherResult.error}` : ""}　语义相似度：{value(item.similarityResult?.score, 3)}</div>{item.modelAnswer && <div className="form-grid"><div className="field"><label className="field-label">人工总体分（0–5）</label><input className="input" type="number" min="0" max="5" step="0.5" value={score} onChange={(event) => setScore(Number(event.target.value))} /></div><div className="field"><label className="field-label">归因</label><select className="input" value={attribution} onChange={(event) => setAttribution(event.target.value)}>{attributionOptions.map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></div><div className="field"><label className="field-label">人工备注</label><input className="input" value={note} onChange={(event) => setNote(event.target.value)} placeholder="说明人工判断依据" /></div></div>} {item.modelAnswer && <button className="secondary-btn" disabled={saving} onClick={() => { setSaving(true); void onReview(item, score, attribution, note).finally(() => setSaving(false)); }}><CheckCircle2 size={16} />{saving ? "保存中" : "确认人工结论"}</button>}</article>;
}
