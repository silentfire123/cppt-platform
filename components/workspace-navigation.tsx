"use client";

import { useEffect, useRef, useState } from "react";
import { Boxes, ChevronDown, ClipboardCheck, Database, FlaskConical, LayoutDashboard, ServerCog } from "lucide-react";
import type { ModelLibraryView } from "../lib/interface-state";

export type WorkspaceModule = "overview" | "datasets" | "training" | "evaluation" | "models" | "inference";

const items = [
  { icon: LayoutDashboard, label: "概览", module: "overview" },
  { icon: Database, label: "数据集", module: "datasets" },
  { icon: FlaskConical, label: "微调任务", module: "training" },
  { icon: ClipboardCheck, label: "模型评测", module: "evaluation" },
  { icon: Boxes, label: "模型管理", module: "models" },
  { icon: ServerCog, label: "推理服务", module: "inference" },
] as const;

export function WorkspaceNavigation({ activeModule, modelView, evaluationEnabled, onNavigate, onModelView }: {
  activeModule: WorkspaceModule;
  modelView: ModelLibraryView;
  evaluationEnabled: boolean;
  onNavigate: (module: WorkspaceModule) => void;
  onModelView: (view: ModelLibraryView) => void;
}) {
  const [expanded, setExpanded] = useState(activeModule === "models");
  const [narrow, setNarrow] = useState(false);
  const [overlayOpen, setOverlayOpen] = useState(false);
  const sidebar = useRef<HTMLElement>(null);
  const modelButton = useRef<HTMLButtonElement>(null);
  const showChildren = expanded && (!narrow || overlayOpen);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 1100px)");
    const update = () => { setNarrow(media.matches); setOverlayOpen(false); };
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    setExpanded(activeModule === "models");
  }, [activeModule]);

  function closeOverlay() {
    setOverlayOpen(false);
    modelButton.current?.focus();
  }

  useEffect(() => {
    if (!overlayOpen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setOverlayOpen(false);
        modelButton.current?.focus();
      }
      if (event.key === "Tab") {
        const buttons = Array.from(sidebar.current?.querySelectorAll<HTMLButtonElement>("button") ?? [])
          .filter((button) => button.getClientRects().length && !button.disabled);
        const first = buttons[0];
        const last = buttons[buttons.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [overlayOpen]);

  function toggleModels() {
    if (narrow) {
      if (overlayOpen) { closeOverlay(); return; }
      setOverlayOpen(true);
      setExpanded(true);
    } else {
      setExpanded(activeModule !== "models" || !expanded);
    }
    if (activeModule !== "models") onNavigate("models");
  }

  return <>
    {overlayOpen && <button className="sidebar-backdrop" aria-label="关闭导航菜单" tabIndex={-1} onClick={closeOverlay} />}
    <aside ref={sidebar} className={`sidebar${overlayOpen ? " sidebar-expanded" : ""}`} role={overlayOpen ? "dialog" : undefined} aria-modal={overlayOpen || undefined} aria-label={overlayOpen ? "导航菜单" : undefined}>
      <div className="brand"><span className="brand-mark">AI</span><span className="brand-name">训推平台</span></div>
      <div className="space-switcher"><span className="space-label">当前业务空间</span><span className="space-name">大模型研发中心⌄</span></div>
      <nav className="nav" aria-label="主导航">
        {items.filter(({ module }) => module !== "evaluation" || evaluationEnabled).map(({ icon: Icon, label, module }) => module === "models" ? (
          <div className="nav-group" key={module}>
            <button ref={modelButton} onClick={toggleModels} className={`nav-item nav-parent${activeModule === "models" ? " active" : ""}`} aria-label={label} aria-expanded={showChildren} aria-controls="model-navigation-children" title={label}>
              <Icon className="nav-icon" size={18} strokeWidth={1.8} aria-hidden="true" /><span>{label}</span>
              <ChevronDown className={`nav-chevron${showChildren ? " expanded" : ""}`} size={15} strokeWidth={1.8} aria-hidden="true" />
            </button>
            <div id="model-navigation-children" className="nav-submenu" hidden={!showChildren}>
              {([{ view: "base", label: "优云公共模型库" }, { view: "trained", label: "训练权重" }] as const).map(({ view, label }) => (
                <button key={view} className={`nav-subitem${activeModule === "models" && modelView === view ? " active" : ""}`} aria-current={activeModule === "models" && modelView === view ? "page" : undefined} onClick={() => { onModelView(view); if (narrow) closeOverlay(); }}>{label}</button>
              ))}
            </div>
          </div>
        ) : (
          <button key={module} onClick={() => { onNavigate(module); if (narrow) { setOverlayOpen(false); } }} className={`nav-item${activeModule === module ? " active" : ""}`} aria-label={label} aria-current={activeModule === module ? "page" : undefined} title={label}>
            <Icon className="nav-icon" size={18} strokeWidth={1.8} aria-hidden="true" /><span>{label}</span>
          </button>
        ))}
      </nav>
    </aside>
  </>;
}
