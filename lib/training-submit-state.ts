export type BackendState = "checking" | "online" | "offline";

export class ApiRequestError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

export function isConnectivityError(error: unknown): boolean {
  return error instanceof ApiRequestError && error.status === null;
}

export function backendStateAfterSubmitError(current: BackendState, error: unknown): BackendState {
  return isConnectivityError(error) ? "offline" : current;
}
