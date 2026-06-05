// Typed wrapper around the VS Code Webview API.
// Falls back gracefully so the app can be developed in a browser with `vite dev`.

interface VSCodeAPI {
  postMessage(message: unknown): void;
  getState(): unknown;
  setState(newState: unknown): void;
}

declare function acquireVsCodeApi(): VSCodeAPI;

let _api: VSCodeAPI | null = null;

function getAPI(): VSCodeAPI | null {
  if (typeof acquireVsCodeApi !== 'undefined') {
    if (!_api) _api = acquireVsCodeApi();
    return _api;
  }
  return null;
}

export function postToExtension(message: unknown): void {
  getAPI()?.postMessage(message);
}

export function isInsideVSCode(): boolean {
  return typeof acquireVsCodeApi !== 'undefined';
}
