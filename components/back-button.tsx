"use client";

import { ArrowLeft } from "lucide-react";

export function BackButton({
  label,
  onClick,
  disabled = false,
  className = "",
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      className={`back-button${className ? ` ${className}` : ""}`}
      disabled={disabled}
      onClick={onClick}
      aria-label={label}
    >
      <ArrowLeft aria-hidden="true" size={16} strokeWidth={1.8} />
      <span>{label}</span>
    </button>
  );
}
