import * as path from 'path';
import { exec, ExecException } from 'child_process';
import * as vscode from 'vscode';
import { BACKEND_URL } from './config.generated';
import type { WebviewMessage } from './types';

function getNonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  return Array.from({ length: 32 }, () => chars[Math.floor(Math.random() * chars.length)]).join('');
}

// Pending local file-write or terminal-execute operations keyed by callId.
type PendingLocalOp =
  | { kind: 'edit'; path: string; proposed: string; workspaceRoot: string }
  | { kind: 'terminal'; command: string; workspaceRoot: string };

export class SidebarProvider implements vscode.WebviewViewProvider {
  static readonly viewType = 'misterpilot.chat';

  private _view?: vscode.WebviewView;
  private _abortController: AbortController | null = null;
  private _pendingLocalOps: Map<string, PendingLocalOp> = new Map();

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

    view.onDidChangeVisibility(() => {
      if (view.visible) {
        this._postWorkspaceRoot(view.webview);
        void this._postApiKeyStatus(view.webview);
      }
    });

    view.webview.onDidReceiveMessage(async (msg: WebviewMessage) => {
      switch (msg.type) {
        case 'ready':
          this._postWorkspaceRoot(view.webview);
          await this._postApiKeyStatus(view.webview);
          break;
        case 'chat':
          await this._handleChat(msg.messages, view.webview, msg.model, msg.mode);
          break;
        case 'stopChat':
          this._abortController?.abort();
          this._pendingLocalOps.clear();
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
    if (key === undefined) return;
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
    webview: vscode.Webview,
    model: string,
    mode: string
  ): Promise<void> {
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? null;
    const apiKey = await this._context.secrets.get('misterpilot.apiKey') ?? '';

    const body: Record<string, unknown> = { messages, model, mode };
    if (workspaceRoot) body.workspace_root = workspaceRoot;

    this._abortController = new AbortController();

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (apiKey) headers['X-Api-Key'] = apiKey;

      const res = await fetch(`${BACKEND_URL}/agent/stream`, {
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
      case 'tool_call_request': {
        // Backend asks extension to execute a tool locally (extension has filesystem access).
        const callId = String(payload.call_id ?? '');
        const tool = String(payload.tool ?? '');
        const args = (payload.args ?? {}) as Record<string, unknown>;
        void this._handleToolCallRequest(callId, tool, args, webview).catch((err) => {
          const msg = err instanceof Error ? err.message : String(err);
          void this._submitToolResult(callId, `Tool handler error: ${msg}`);
        });
        break;
      }
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
      case 'cost':
        webview.postMessage({ type: 'cost', usd: Number(payload.usd ?? 0) });
        break;
    }
  }

  // ── local tool execution ──────────────────────────────────────────────────

  private async _handleToolCallRequest(
    callId: string,
    tool: string,
    args: Record<string, unknown>,
    webview: vscode.Webview
  ): Promise<void> {
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? null;

    switch (tool) {
      case 'read_file': {
        const content = await this._localReadFile(workspaceRoot, String(args.path ?? ''));
        await this._submitToolResult(callId, content);
        break;
      }
      case 'list_files': {
        const content = await this._localListFiles(workspaceRoot);
        await this._submitToolResult(callId, content);
        break;
      }
      case 'search_code': {
        const content = await this._localSearchCode(
          workspaceRoot,
          String(args.query ?? ''),
          args.file_pattern ? String(args.file_pattern) : undefined,
          Boolean(args.case_sensitive ?? false)
        );
        await this._submitToolResult(callId, content);
        break;
      }
      case 'write_file': {
        const filePath = args.path ? String(args.path) : '';
        if (!filePath) { await this._submitToolResult(callId, 'Error: missing required parameter "path"'); break; }
        await this._localStagePendingEdit(callId, filePath, String(args.content ?? ''), workspaceRoot, webview);
        break;
      }
      case 'replace_in_file': {
        const filePath = args.path ? String(args.path) : '';
        if (!filePath) { await this._submitToolResult(callId, 'Error: missing required parameter "path"'); break; }
        await this._localStageReplaceEdit(callId, filePath, String(args.old_text ?? ''), String(args.new_text ?? ''), workspaceRoot, webview);
        break;
      }
      case 'execute_terminal':
        await this._localStageTerminal(callId, String(args.command ?? ''), workspaceRoot, webview);
        break;
      default:
        await this._submitToolResult(callId, `Unknown tool: ${tool}`);
    }
  }

  private async _localReadFile(workspaceRoot: string | null, filePath: string): Promise<string> {
    if (!workspaceRoot) return 'No workspace open';
    const abs = this._safeLocalPath(workspaceRoot, filePath);
    if (!abs) return 'Error: path traversal attempt blocked';
    try {
      const bytes = await vscode.workspace.fs.readFile(vscode.Uri.file(abs));
      const raw = Buffer.from(bytes).toString('utf-8');
      const lines = raw.split('\n');
      const numbered = lines.slice(0, 150).map((l, i) => `${i + 1}: ${l}`).join('\n');
      const suffix = lines.length > 150 ? `\n... (${lines.length - 150} more lines)` : '';
      return numbered + suffix;
    } catch (err) {
      return `Error reading file: ${err instanceof Error ? err.message : String(err)}`;
    }
  }

  private async _localListFiles(workspaceRoot: string | null): Promise<string> {
    if (!workspaceRoot) return 'No workspace open';
    try {
      const pattern = new vscode.RelativePattern(workspaceRoot, '**/*');
      const exclude = '{**/node_modules/**,**/.git/**,**/dist/**,**/__pycache__/**,**/.venv/**,**/out/**}';
      const files = await vscode.workspace.findFiles(pattern, exclude, 500);
      const paths = files
        .map(f => path.relative(workspaceRoot, f.fsPath))
        .sort();
      const body = paths.slice(0, 300).join('\n');
      const suffix = paths.length > 300 ? `\n... (${paths.length - 300} more files)` : '';
      return body + suffix || '(empty workspace)';
    } catch (err) {
      return `Error listing files: ${err instanceof Error ? err.message : String(err)}`;
    }
  }

  private async _localSearchCode(
    workspaceRoot: string | null,
    query: string,
    filePattern: string | undefined,
    caseSensitive: boolean
  ): Promise<string> {
    if (!workspaceRoot) return 'No workspace open';
    if (!query) return 'Error: empty search query';
    try {
      // Escape query for use as a fixed string
      const escaped = query.replace(/[\\'"]/g, '\\$&');
      const caseFlag = caseSensitive ? '' : '-i';
      const globArg = filePattern ? `--include="${filePattern}"` : '';
      // Try ripgrep first, fall back to grep
      const rgCmd = `rg --fixed-strings ${caseFlag} -n --max-count 50 ${globArg} ${JSON.stringify(query)} .`;
      const grepCmd = `grep -r --fixed-strings ${caseFlag} -n -m 50 ${globArg} ${JSON.stringify(escaped)} .`;

      let output = '';
      try {
        const result = await this._execLocal(rgCmd, workspaceRoot);
        if (result.exitCode === 0 || result.stdout) {
          output = result.stdout;
        }
      } catch {
        // rg not available, try grep
        try {
          const result = await this._execLocal(grepCmd, workspaceRoot);
          output = result.stdout;
        } catch (grepErr) {
          return `Error searching: ${grepErr instanceof Error ? grepErr.message : String(grepErr)}`;
        }
      }

      if (!output.trim()) return `No matches found for: ${JSON.stringify(query)}`;
      const lines = output.trim().split('\n').slice(0, 50);
      return `${lines.length} match(es):\n` + lines.join('\n');
    } catch (err) {
      return `Error searching: ${err instanceof Error ? err.message : String(err)}`;
    }
  }

  private async _localStagePendingEdit(
    callId: string,
    filePath: string,
    proposed: string,
    workspaceRoot: string | null,
    webview: vscode.Webview
  ): Promise<void> {
    if (!workspaceRoot) { await this._submitToolResult(callId, 'No workspace open'); return; }
    const abs = this._safeLocalPath(workspaceRoot, filePath);
    if (!abs) { await this._submitToolResult(callId, 'Error: path traversal attempt blocked'); return; }

    let original = '';
    let isNewFile = true;
    try {
      const bytes = await vscode.workspace.fs.readFile(vscode.Uri.file(abs));
      original = Buffer.from(bytes).toString('utf-8');
      isNewFile = false;
    } catch { /* file doesn't exist yet */ }

    const diff = isNewFile ? '' : this._computeUnifiedDiff(original, proposed, filePath);
    this._pendingLocalOps.set(callId, { kind: 'edit', path: filePath, proposed, workspaceRoot });
    webview.postMessage({ type: 'pendingEdit', editId: callId, path: filePath, diff, original, proposed, isNewFile });
  }

  private async _localStageReplaceEdit(
    callId: string,
    filePath: string,
    oldText: string,
    newText: string,
    workspaceRoot: string | null,
    webview: vscode.Webview
  ): Promise<void> {
    if (!workspaceRoot) { await this._submitToolResult(callId, 'No workspace open'); return; }
    const abs = this._safeLocalPath(workspaceRoot, filePath);
    if (!abs) { await this._submitToolResult(callId, 'Error: path traversal attempt blocked'); return; }

    let original: string;
    try {
      const bytes = await vscode.workspace.fs.readFile(vscode.Uri.file(abs));
      original = Buffer.from(bytes).toString('utf-8');
    } catch {
      await this._submitToolResult(callId, `File not found: ${filePath}`);
      return;
    }

    if (!original.includes(oldText)) {
      await this._submitToolResult(callId, `Text to replace not found in ${filePath}`);
      return;
    }

    const proposed = original.replace(oldText, newText);
    const diff = this._computeUnifiedDiff(original, proposed, filePath);
    this._pendingLocalOps.set(callId, { kind: 'edit', path: filePath, proposed, workspaceRoot });
    webview.postMessage({ type: 'pendingEdit', editId: callId, path: filePath, diff, original, proposed, isNewFile: false });
  }

  private async _localStageTerminal(
    callId: string,
    command: string,
    workspaceRoot: string | null,
    webview: vscode.Webview
  ): Promise<void> {
    if (!workspaceRoot) { await this._submitToolResult(callId, 'No workspace open'); return; }
    if (!command.trim()) { await this._submitToolResult(callId, 'Error: empty command'); return; }

    this._pendingLocalOps.set(callId, { kind: 'terminal', command, workspaceRoot });
    webview.postMessage({ type: 'terminalPending', id: callId, command });
  }

  private async _submitToolResult(callId: string, content: string): Promise<void> {
    try {
      await fetch(`${BACKEND_URL}/agent/tool_result`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ call_id: callId, content }),
      });
    } catch { /* agent will timeout after 5 min */ }
  }

  private _safeLocalPath(workspaceRoot: string, filePath: string): string | null {
    const abs = path.isAbsolute(filePath)
      ? path.normalize(filePath)
      : path.normalize(path.join(workspaceRoot, filePath));
    const root = path.normalize(workspaceRoot);
    if (!abs.startsWith(root + path.sep)) return null;
    return abs;
  }

  private _computeUnifiedDiff(original: string, proposed: string, filename: string): string {
    const a = original.split('\n');
    const b = proposed.split('\n');
    const m = a.length, n = b.length;
    const CONTEXT = 3;

    const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 1; i <= m; i++)
      for (let j = 1; j <= n; j++)
        dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);

    type Edit = { op: ' ' | '+' | '-'; line: string };
    const edits: Edit[] = [];
    let i = m, j = n;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) { edits.unshift({ op: ' ', line: a[i - 1] }); i--; j--; }
      else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) { edits.unshift({ op: '+', line: b[j - 1] }); j--; }
      else { edits.unshift({ op: '-', line: a[i - 1] }); i--; }
    }

    const changed = edits.map((e, k) => e.op !== ' ' ? k : -1).filter(k => k >= 0);
    if (!changed.length) return '';

    const out: string[] = [`--- ${filename}`, `+++ ${filename}`];
    const hunks: Array<[number, number]> = [];
    let hs = Math.max(0, changed[0] - CONTEXT), he = Math.min(edits.length - 1, changed[0] + CONTEXT);
    for (let k = 1; k < changed.length; k++) {
      const ns = Math.max(0, changed[k] - CONTEXT);
      if (ns <= he + 1) { he = Math.min(edits.length - 1, changed[k] + CONTEXT); }
      else { hunks.push([hs, he]); hs = ns; he = Math.min(edits.length - 1, changed[k] + CONTEXT); }
    }
    hunks.push([hs, he]);

    for (const [s, e] of hunks) {
      let oldL = 1, newL = 1;
      for (let k = 0; k < s; k++) { if (edits[k].op !== '+') oldL++; if (edits[k].op !== '-') newL++; }
      let oldC = 0, newC = 0;
      for (let k = s; k <= e; k++) { if (edits[k].op !== '+') oldC++; if (edits[k].op !== '-') newC++; }
      out.push(`@@ -${oldL},${oldC} +${newL},${newC} @@`);
      for (let k = s; k <= e; k++) out.push(`${edits[k].op}${edits[k].line}`);
    }
    return out.join('\n');
  }

  private _execLocal(command: string, cwd: string): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    return new Promise((resolve) => {
      exec(command, { cwd, timeout: 30000, maxBuffer: 50 * 1024 }, (err: ExecException | null, stdout: string, stderr: string) => {
        resolve({
          stdout: stdout.slice(0, 50000),
          stderr: stderr.slice(0, 50000),
          exitCode: (err && typeof err.code === 'number') ? err.code : (err ? 1 : 0),
        });
      });
    });
  }

  // ── file edit operations ──────────────────────────────────────────────────

  private async _applyEdit(editId: string, webview: vscode.Webview): Promise<void> {
    // Local pending op (agent-driven, remote backend scenario)
    const localOp = this._pendingLocalOps.get(editId);
    if (localOp && localOp.kind === 'edit') {
      this._pendingLocalOps.delete(editId);
      try {
        const abs = this._safeLocalPath(localOp.workspaceRoot, localOp.path);
        if (!abs) throw new Error('Path traversal attempt blocked');
        await vscode.workspace.fs.createDirectory(vscode.Uri.file(path.dirname(abs)));
        await vscode.workspace.fs.writeFile(vscode.Uri.file(abs), Buffer.from(localOp.proposed, 'utf-8'));
        webview.postMessage({ type: 'editApplied', editId, path: localOp.path });
        vscode.window.showInformationMessage(`MisterPilot: Applied → ${localOp.path}`);
        await this._submitToolResult(editId, `File written: ${localOp.path}`);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        webview.postMessage({ type: 'editError', editId, message });
        vscode.window.showErrorMessage(`MisterPilot apply failed: ${message}`);
        await this._submitToolResult(editId, `Error writing file: ${message}`);
      }
      return;
    }

    // Fallback: direct backend apply (standalone usage)
    try {
      const res = await fetch(`${BACKEND_URL}/edit/apply`, {
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
    const localOp = this._pendingLocalOps.get(editId);
    if (localOp && localOp.kind === 'edit') {
      this._pendingLocalOps.delete(editId);
      webview.postMessage({ type: 'editRejected', editId, path: localOp.path });
      await this._submitToolResult(editId, `Edit rejected by user: ${localOp.path}`);
      return;
    }

    try {
      const res = await fetch(`${BACKEND_URL}/edit/reject`, {
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
    const localOp = this._pendingLocalOps.get(id);
    if (localOp && localOp.kind === 'terminal') {
      this._pendingLocalOps.delete(id);

      if (!approved) {
        webview.postMessage({
          type: 'terminalResult',
          id,
          command: localOp.command,
          approved: false,
          stdout: '',
          stderr: '',
          exitCode: -1,
          durationMs: 0,
          timedOut: false,
          error: null,
        });
        await this._submitToolResult(id, `Command denied by user: ${localOp.command}`);
        return;
      }

      const t0 = Date.now();
      try {
        const { stdout, stderr, exitCode } = await this._execLocal(localOp.command, localOp.workspaceRoot);
        const durationMs = Date.now() - t0;
        webview.postMessage({
          type: 'terminalResult',
          id,
          command: localOp.command,
          approved: true,
          stdout,
          stderr,
          exitCode,
          durationMs,
          timedOut: false,
          error: null,
        });
        const parts: string[] = [];
        if (stdout) parts.push(`stdout:\n${stdout}`);
        if (stderr) parts.push(`stderr:\n${stderr}`);
        parts.push(`exit code: ${exitCode}`);
        await this._submitToolResult(id, parts.join('\n') || '(no output)');
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        webview.postMessage({ type: 'terminalError', id, command: localOp.command, message });
        await this._submitToolResult(id, `Command failed: ${message}`);
      }
      return;
    }

    // Fallback: delegate to backend (standalone usage)
    let command = '';
    try {
      const info = await fetch(`${BACKEND_URL}/terminal/pending/${id}`);
      if (info.ok) {
        const d = await info.json() as { command: string };
        command = d.command;
      }
    } catch { /* best-effort */ }

    try {
      const res = await fetch(`${BACKEND_URL}/terminal/execute`, {
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
