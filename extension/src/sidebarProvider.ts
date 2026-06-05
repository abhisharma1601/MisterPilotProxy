import * as vscode from 'vscode';
import type { WebviewMessage } from './types';

function getNonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  return Array.from({ length: 32 }, () => chars[Math.floor(Math.random() * chars.length)]).join('');
}

export class SidebarProvider implements vscode.WebviewViewProvider {
  static readonly viewType = 'misterpilot.chat';

  private _view?: vscode.WebviewView;
  private _abortController: AbortController | null = null;

  constructor(private readonly _context: vscode.ExtensionContext) {}

  resolveWebviewView(
    view: vscode.WebviewView,
    _ctx: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = view;

    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._context.extensionUri],
    };

    view.webview.html = this._buildHtml(view.webview);

    view.webview.onDidReceiveMessage(async (msg: WebviewMessage) => {
      switch (msg.type) {
        case 'ready':
          this._postWorkspaceRoot(view.webview);
          await this._postApiKeyStatus(view.webview);
          break;
        case 'chat':
          await this._handleChat(msg.messages, view.webview);
          break;
        case 'stopChat':
          this._abortController?.abort();
          break;
        case 'setApiKey':
          await this._promptAndSaveApiKey(view.webview);
          break;
        case 'applyEdit':
          await this._applyEdit(msg.editId, view.webview);
          break;
        case 'rejectEdit':
          await this._rejectEdit(msg.editId, view.webview);
          break;
        case 'viewDiff':
          await this._openDiffEditor(msg.path, msg.original, msg.proposed);
          break;
        case 'approveTerminal':
          await this._executeTerminal(msg.id, msg.approved, view.webview);
          break;
      }
    });
  }

  // ── helpers ──────────────────────────────────────────────────────────────

  private _backendUrl(): string {
    return vscode.workspace
      .getConfiguration('misterpilot')
      .get<string>('backendUrl', 'http://localhost:8000');
  }

  private _postWorkspaceRoot(webview: vscode.Webview): void {
    const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? null;
    webview.postMessage({ type: 'workspaceRoot', path: root });
  }

  private async _postApiKeyStatus(webview: vscode.Webview): Promise<void> {
    const key = await this._context.secrets.get('misterpilot.apiKey');
    webview.postMessage({ type: 'apiKeyStatus', isSet: !!key });
  }

  private async _promptAndSaveApiKey(webview: vscode.Webview): Promise<void> {
    const key = await vscode.window.showInputBox({
      prompt: 'Enter your API key for the LLM backend',
      placeHolder: 'sk-...',
      password: true,
      ignoreFocusOut: true,
    });
    if (key === undefined) return; // user cancelled
    if (key === '') {
      await this._context.secrets.delete('misterpilot.apiKey');
      webview.postMessage({ type: 'apiKeyStatus', isSet: false });
      vscode.window.showInformationMessage('MisterPilot: API key cleared.');
    } else {
      await this._context.secrets.store('misterpilot.apiKey', key);
      webview.postMessage({ type: 'apiKeyStatus', isSet: true });
      vscode.window.showInformationMessage('MisterPilot: API key saved.');
    }
  }

  // ── chat + streaming ──────────────────────────────────────────────────────

  private async _handleChat(
    messages: Record<string, unknown>[],
    webview: vscode.Webview
  ): Promise<void> {
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? null;
    const apiKey = await this._context.secrets.get('misterpilot.apiKey') ?? '';

    const body: Record<string, unknown> = { messages };
    if (workspaceRoot) body.workspace_root = workspaceRoot;

    this._abortController = new AbortController();

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (apiKey) headers['X-Api-Key'] = apiKey;

      const res = await fetch(`${this._backendUrl()}/agent/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: this._abortController.signal,
      });

      if (!res.ok) throw new Error(`Backend ${res.status}: ${res.statusText}`);
      if (!res.body) throw new Error('Empty response body from backend');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          let payload: Record<string, unknown>;
          try { payload = JSON.parse(raw); } catch { continue; }
          this._dispatchSsePayload(payload, webview);
        }
      }

      webview.postMessage({ type: 'done' });
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        webview.postMessage({ type: 'done' });
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      webview.postMessage({ type: 'error', message });
      webview.postMessage({ type: 'done' });
    } finally {
      this._abortController = null;
    }
  }

  private _dispatchSsePayload(
    payload: Record<string, unknown>,
    webview: vscode.Webview
  ): void {
    switch (payload.type) {
      case 'chunk':
        webview.postMessage({ type: 'chunk', content: payload.content });
        break;
      case 'error':
        webview.postMessage({ type: 'error', message: payload.message ?? 'Unknown error' });
        break;
      case 'done':
        webview.postMessage({ type: 'done' });
        break;
      case 'pending_edit':
        webview.postMessage({
          type: 'pendingEdit',
          editId: payload.id,
          path: payload.path,
          diff: payload.diff,
          original: payload.original,
          proposed: payload.proposed,
          isNewFile: payload.is_new_file ?? false,
        });
        break;
      case 'pending_terminal':
        webview.postMessage({
          type: 'terminalPending',
          id: String(payload.id ?? ''),
          command: String(payload.command ?? ''),
        });
        break;
      case 'tool_call':
        webview.postMessage({
          type: 'toolCall',
          id: String(payload.id ?? ''),
          tool: String(payload.tool ?? ''),
          args: (payload.args ?? {}) as Record<string, unknown>,
        });
        break;
      case 'tool_result':
        webview.postMessage({
          type: 'toolResult',
          id: String(payload.id ?? ''),
          tool: String(payload.tool ?? ''),
          content: String(payload.content ?? ''),
        });
        break;
    }
  }

  // ── file edit operations ──────────────────────────────────────────────────

  private async _applyEdit(editId: string, webview: vscode.Webview): Promise<void> {
    try {
      const res = await fetch(`${this._backendUrl()}/edit/apply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: editId }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` })) as { detail?: string };
        throw new Error(err.detail ?? String(res.status));
      }

      const data = await res.json() as { path: string };
      webview.postMessage({ type: 'editApplied', editId, path: data.path });
      vscode.window.showInformationMessage(`MisterPilot: Applied → ${data.path}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      webview.postMessage({ type: 'editError', editId, message });
      vscode.window.showErrorMessage(`MisterPilot apply failed: ${message}`);
    }
  }

  private async _rejectEdit(editId: string, webview: vscode.Webview): Promise<void> {
    try {
      const res = await fetch(`${this._backendUrl()}/edit/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: editId }),
      });
      const data = res.ok ? await res.json() as { path: string } : { path: '' };
      webview.postMessage({ type: 'editRejected', editId, path: data.path });
    } catch {
      webview.postMessage({ type: 'editRejected', editId, path: '' });
    }
  }

  private async _openDiffEditor(
    path: string,
    original: string,
    proposed: string
  ): Promise<void> {
    const ext = path.split('.').pop() ?? 'txt';
    const [origDoc, propDoc] = await Promise.all([
      vscode.workspace.openTextDocument({ content: original, language: ext }),
      vscode.workspace.openTextDocument({ content: proposed, language: ext }),
    ]);
    await vscode.commands.executeCommand(
      'vscode.diff',
      origDoc.uri,
      propDoc.uri,
      `MisterPilot: ${path} (proposed)`
    );
  }

  // ── terminal execution ────────────────────────────────────────────────────

  private async _executeTerminal(
    id: string,
    approved: boolean,
    webview: vscode.Webview
  ): Promise<void> {
    let command = '';
    try {
      const info = await fetch(`${this._backendUrl()}/terminal/pending/${id}`);
      if (info.ok) {
        const d = await info.json() as { command: string };
        command = d.command;
      }
    } catch { /* best-effort */ }

    try {
      const res = await fetch(`${this._backendUrl()}/terminal/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, approved }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` })) as { detail?: string };
        throw new Error(err.detail ?? String(res.status));
      }

      const data = await res.json() as {
        approved: boolean;
        stdout: string;
        stderr: string;
        exit_code: number;
        duration_ms: number;
        timed_out: boolean;
        error: string | null;
      };

      webview.postMessage({
        type: 'terminalResult',
        id,
        command: command || data.approved.toString(),
        approved: data.approved,
        stdout: data.stdout,
        stderr: data.stderr,
        exitCode: data.exit_code,
        durationMs: data.duration_ms,
        timedOut: data.timed_out,
        error: data.error,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      webview.postMessage({ type: 'terminalError', id, command, message });
    }
  }

  // ── webview HTML ──────────────────────────────────────────────────────────

  private _buildHtml(webview: vscode.Webview): string {
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, 'webview-dist', 'assets', 'index.js')
    );
    const nonce = getNonce();

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none';
             style-src 'unsafe-inline';
             script-src 'nonce-${nonce}';
             img-src ${webview.cspSource} data:;
             font-src ${webview.cspSource};">
  <title>MisterPilot</title>
</head>
<body>
  <div id="root"></div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }
}
