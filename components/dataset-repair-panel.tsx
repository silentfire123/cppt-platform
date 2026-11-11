"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { apiUrl } from "../lib/api-config";
import { formatApiError } from "../lib/api-error";
import { Dataset, DatasetValidationIssue, DatasetValidationReport } from "../lib/types";

type RepairResponse = {
  dataset?: Dataset;
  stats?: Record<string, number>;
  report?: DatasetValidationReport;
  canApply?: boolean;
  createdNewVersion?: boolean;
};

type RepairRow = { line: number; record: Record<string, unknown> | null; raw: string };

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), { ...options, credentials: "include", cache: "no-store" });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(formatApiError(body, `请求失败（${response.status}）`));
  return body as T;
}

function issueLabel(issue: DatasetValidationIssue) {
  const line = issue.line ? `第 ${issue.line} 行` : "文件级";
  return `${line}${issue.path ? ` · ${issue.path}` : ""}`;
}

export function DatasetRepairPanel({
  dataset,
  onUpdated,
}: {
  dataset: Dataset;
  onUpdated: (dataset: Dataset) => void;
}) {
  const [report, setReport] = useState<DatasetValidationReport>(dataset.validationReport ?? {});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [rows, setRows] = useState<RepairRow[]>([]);
  const [editedRows, setEditedRows] = useState<Record<number, Record<string, unknown>>>({});
  const [editedRawRows, setEditedRawRows] = useState<Record<number, string>>({});
  const [preview, setPreview] = useState<RepairResponse | null>(null);
  const [focusedLine, setFocusedLine] = useState<number | null>(null);
  const rowRefs = useRef<Record<number, HTMLDivElement | null>>({});

  useEffect(() => {
    let active = true;
    setError("");
    void request<{ report: DatasetValidationReport }>(`/datasets/${dataset.id}/validation`)
      .then((result) => { if (active) setReport(result.report); })
      .catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : "读取校验报告失败。"); });
    return () => { active = false; };
  }, [dataset.id]);

  useEffect(() => {
    void request<{ rows: RepairRow[] }>(`/datasets/${dataset.id}/validation/rows`)
      .then((result) => setRows(result.rows))
      .catch((reason) => setError(reason instanceof Error ? reason.message : "读取错误行失败。"));
  }, [dataset.id]);

  useEffect(() => {
    if (focusedLine === null || !rowRefs.current[focusedLine]) return;
    const node = rowRefs.current[focusedLine];
    node?.scrollIntoView({ behavior: "smooth", block: "center" });
    const editor = node?.querySelector("textarea, select") as HTMLElement | null;
    editor?.focus();
  }, [focusedLine, rows]);

  const errors = useMemo(() => report.errors ?? [], [report.errors]);
  const warnings = useMemo(() => report.warnings ?? [], [report.warnings]);

  function jumpToLine(line: number | null | undefined) {
    if (line == null) return;
    setFocusedLine(line);
  }

  function editPayload() {
    return {
      mode: "manual" as const,
      edits: [
        ...Object.entries(editedRows).map(([line, record]) => ({ line: Number(line), record })),
        ...Object.entries(editedRawRows).map(([line, raw]) => ({ line: Number(line), raw })),
      ],
    };
  }

  async function previewRepair() {
    setLoading(true);
    setError("");
    try {
      const result = await request<RepairResponse>(`/datasets/${dataset.id}/repair/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editPayload()),
      });
      setPreview(result);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "生成修复预览失败。");
    } finally { setLoading(false); }
  }

  async function applyRepair() {
    setLoading(true);
    setError("");
    try {
      const result = await request<RepairResponse>(`/datasets/${dataset.id}/repair`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editPayload()),
      });
      if (result.dataset) onUpdated(result.dataset);
      setReport(result.report ?? {});
      setPreview(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "应用修复失败。");
    } finally { setLoading(false); }
  }

  return <div className="dataset-repair-panel">
    <div className="repair-summary">
      <div><span>检查行数</span><strong>{report.checkedRows ?? "—"}</strong></div>
      <div><span>错误行数</span><strong className={errors.length ? "danger-text" : "success-text"}>{report.invalidRows ?? errors.length}</strong></div>
      <div><span>需人工确认</span><strong>{errors.length}</strong></div>
    </div>
    {errors.length > 0 ? <div className="repair-issues">
      <h4>错误详情</h4>
      {errors.map((issue, index) => <div className="repair-issue" key={`${issue.code}-${issue.line}-${index}`}>
        <div>
          <button type="button" className="repair-issue-jump" onClick={() => jumpToLine(issue.line)} disabled={issue.line == null}>
            <strong>{issueLabel(issue)}</strong>
          </button>
          <span className="repair-code">{issue.code}</span>
          {issue.line != null && <small className="repair-jump-hint">点击定位到编辑行</small>}
        </div>
        <p>{issue.message}</p>
        {issue.suggestion && <small>{issue.suggestion}</small>}
      </div>)}
    </div> : <p className="success-text">当前版本通过结构校验。</p>}
    {warnings.length > 0 && <p className="validation-warning">另有 {warnings.length} 条警告，通常不会阻止发布。</p>}
    <div className="repair-row-editor">
      <p className="repair-editor-help">这里只能手动编辑校验报告指出的行。系统不会自动转换 role、清理文本或删除空行；修改后先预览，再确认保存。</p>
      {rows.length === 0 && <div className="empty">当前没有可编辑的结构化错误行。</div>}
      {rows.map((row) => {
        const current = editedRows[row.line] ?? row.record ?? {};
        const messages = Array.isArray(current.messages) ? current.messages as Array<Record<string, unknown>> : [];
        const update = (next: Record<string, unknown>) => setEditedRows((items) => ({ ...items, [row.line]: next }));
        return <div className={`repair-row-card${focusedLine === row.line ? " focused" : ""}`} key={row.line} ref={(node) => { rowRefs.current[row.line] = node; }}>
          <strong>第 {row.line} 行</strong>
          {messages.length > 0 ? messages.map((message, index) => <div className="repair-message-editor" key={index}>
            <select value={String(message.role ?? "")} onChange={(event) => { const next = messages.map((item, itemIndex) => itemIndex === index ? { ...item, role: event.target.value } : item); update({ ...current, messages: next }); }}>
              <option value="system">system</option><option value="user">user</option><option value="assistant">assistant</option>
            </select>
            <textarea value={String(message.content ?? "")} onChange={(event) => { const next = messages.map((item, itemIndex) => itemIndex === index ? { ...item, content: event.target.value } : item); update({ ...current, messages: next }); }} />
          </div>) : <textarea className="repair-json-fallback" value={editedRawRows[row.line] ?? row.raw} onChange={(event) => setEditedRawRows((items) => ({ ...items, [row.line]: event.target.value }))} />}
        </div>;
      })}
    </div>
    {error && <div className="form-error">{error}</div>}
    <div className="repair-actions">
      <button type="button" className="secondary-btn" onClick={() => void previewRepair()} disabled={loading || (!Object.keys(editedRows).length && !Object.keys(editedRawRows).length)}>{loading ? "处理中..." : "预览修复结果"}</button>
      {preview && <button type="button" className="primary-btn" onClick={() => void applyRepair()} disabled={loading || preview.canApply === false}>确认应用修复</button>}
    </div>
    {preview && <div className={`repair-preview ${preview.canApply ? "preview-ok" : "preview-failed"}`}>
      <strong>{preview.canApply ? "预览通过，可以生成新版本" : "预览后仍有错误，暂不能应用"}</strong>
      {preview.stats && <span>人工修改 {preview.stats.repairedRows ?? 0} 行，未修改错误行 {preview.stats.unchangedInvalidRows ?? 0} 行；原始空行不会被自动删除。</span>}
    </div>}
  </div>;
}
