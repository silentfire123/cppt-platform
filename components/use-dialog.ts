import { RefObject, useEffect, useRef } from "react";

const FOCUSABLE =
  'a[href],button:not([disabled]),textarea:not([disabled]),input:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])';

let openDialogs = 0;

export function useDialog<T extends HTMLElement>(
  open: boolean,
  onClose: () => void,
  options: { closeOnEscape?: boolean } = {},
): RefObject<T> {
  const ref = useRef<T | null>(null);
  const restoreFocus = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const closeOnEscape = options.closeOnEscape ?? true;

  useEffect(() => {
    if (!open) return;
    restoreFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    openDialogs += 1;
    const node = ref.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const focusFirst = () => {
      const target = node?.querySelector<HTMLElement>(FOCUSABLE);
      (target ?? node)?.focus();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      // Close the dialog that currently owns focus, including nested dialogs.
      if (event.key === "Escape" && closeOnEscape && node?.contains(document.activeElement)) {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (event.key !== "Tab" || !node) return;
      const items = Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE));
      if (!items.length) {
        event.preventDefault();
        node.focus();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    const frame = window.requestAnimationFrame(focusFirst);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("keydown", onKeyDown);
      openDialogs = Math.max(0, openDialogs - 1);
      document.body.style.overflow = previousOverflow;
      restoreFocus.current?.focus();
    };
  }, [open, closeOnEscape]);

  return ref;
}
