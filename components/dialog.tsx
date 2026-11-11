"use client";

import { ReactNode, useCallback } from "react";
import { useDialog } from "./use-dialog";

type DialogProps = {
  open: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
  className?: string;
  backdropClassName?: string;
  closeOnBackdrop?: boolean;
};

export function Dialog({ open, title, onClose, children, className, backdropClassName = "modal-backdrop", closeOnBackdrop = true }: DialogProps) {
  const handleClose = useCallback(() => onClose(), [onClose]);
  const ref = useDialog<HTMLDivElement>(open, handleClose);
  if (!open) return null;
  const titleId = `dialog-title-${title.replace(/[^a-z0-9\u4e00-\u9fff]+/gi, "-")}`;
  return <div className={backdropClassName} role="presentation" onMouseDown={(event) => { if (closeOnBackdrop && event.target === event.currentTarget) onClose(); }}>
    <div ref={ref} className={className ?? "modal"} role="dialog" aria-modal="true" aria-labelledby={titleId} tabIndex={-1}>
      <h2 id={titleId} className="sr-only">{title}</h2>
      {children}
    </div>
  </div>;
}
