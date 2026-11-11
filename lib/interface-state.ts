/** Restore the last in-app module only on a browser reload, not on a fresh entry from Control. */
export function shouldRestoreInterfaceState(navigationType: string | undefined): boolean {
  return navigationType === "reload";
}
export type ModelLibraryView = "base" | "trained";

export function restoreModelLibraryView(value: unknown): ModelLibraryView {
  return value === "trained" ? "trained" : "base";
}

export const TRAINING_DRAFT_STORAGE_KEY = "xuntui-training-create-draft";
export const INFERENCE_DRAFT_STORAGE_KEY = "xuntui-inference-create-draft";

export function readSessionDraft<T>(key: string): T | null {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

export function writeSessionDraft(key: string, value: unknown): void {
  try {
    sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Session storage is best effort; never block task creation.
  }
}

export function clearSessionDraft(key: string): void {
  try {
    sessionStorage.removeItem(key);
  } catch {
    // Ignore unavailable storage.
  }
}
