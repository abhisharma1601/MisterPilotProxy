import * as crypto from 'crypto';
import * as vscode from 'vscode';

// Lightweight per-chat metadata kept in index.json for the switcher list.
export interface ChatMeta {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messageCount: number;
}

// The render/continue snapshot a chat carries. uiMessages restores the visual
// cards; llmHistory (OpenAI format) lets the conversation continue.
export interface ChatSnapshot {
  uiMessages: unknown[];
  llmHistory: Record<string, unknown>[];
  model: string;
  mode: string;
  sessionCost: number;
  sessionCostInr: number;
}

export interface StoredChat extends ChatSnapshot, ChatMeta {}

// Cap on retained chats per workspace; oldest beyond this are pruned on save.
const MAX_CHATS = 100;

/**
 * Durable chat storage in the extension's private globalStorage, namespaced per
 * workspace. Layout:
 *   <globalStorage>/chats/<workspaceHash>/index.json   — ChatMeta[]
 *   <globalStorage>/chats/<workspaceHash>/<chatId>.json — StoredChat
 */
export class ChatStore {
  private readonly _wsKey: string;
  private readonly _activeKey: string;

  constructor(private readonly _context: vscode.ExtensionContext) {
    const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? 'no-workspace';
    this._wsKey = crypto.createHash('sha1').update(root).digest('hex').slice(0, 16);
    this._activeKey = `misterpilot.activeChatId.${this._wsKey}`;
  }

  // ── paths ──────────────────────────────────────────────────────────────────

  private _dir(): vscode.Uri {
    return vscode.Uri.joinPath(this._context.globalStorageUri, 'chats', this._wsKey);
  }

  private _indexUri(): vscode.Uri {
    return vscode.Uri.joinPath(this._dir(), 'index.json');
  }

  private _chatUri(id: string): vscode.Uri {
    return vscode.Uri.joinPath(this._dir(), `${id}.json`);
  }

  private async _ensureDir(): Promise<void> {
    await vscode.workspace.fs.createDirectory(this._dir());
  }

  private async _readJson<T>(uri: vscode.Uri, fallback: T): Promise<T> {
    try {
      const bytes = await vscode.workspace.fs.readFile(uri);
      return JSON.parse(Buffer.from(bytes).toString('utf-8')) as T;
    } catch {
      return fallback;
    }
  }

  private async _writeJson(uri: vscode.Uri, value: unknown): Promise<void> {
    await this._ensureDir();
    await vscode.workspace.fs.writeFile(uri, Buffer.from(JSON.stringify(value), 'utf-8'));
  }

  // ── active chat tracking ─────────────────────────────────────────────────

  getActiveChatId(): string | null {
    return this._context.workspaceState.get<string>(this._activeKey) ?? null;
  }

  async setActiveChatId(id: string | null): Promise<void> {
    await this._context.workspaceState.update(this._activeKey, id ?? undefined);
  }

  // ── index ────────────────────────────────────────────────────────────────

  async listChats(): Promise<ChatMeta[]> {
    const index = await this._readJson<ChatMeta[]>(this._indexUri(), []);
    return index.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  private async _upsertIndex(meta: ChatMeta): Promise<void> {
    const index = await this._readJson<ChatMeta[]>(this._indexUri(), []);
    const next = index.filter((c) => c.id !== meta.id);
    next.push(meta);
    await this._writeJson(this._indexUri(), next);
  }

  private async _removeFromIndex(id: string): Promise<void> {
    const index = await this._readJson<ChatMeta[]>(this._indexUri(), []);
    await this._writeJson(this._indexUri(), index.filter((c) => c.id !== id));
  }

  // Keep only the MAX_CHATS most-recently-updated chats; delete the rest.
  private async _pruneToLimit(): Promise<void> {
    const index = await this._readJson<ChatMeta[]>(this._indexUri(), []);
    if (index.length <= MAX_CHATS) return;
    const sorted = [...index].sort((a, b) => b.updatedAt - a.updatedAt);
    const keep = sorted.slice(0, MAX_CHATS);
    const drop = sorted.slice(MAX_CHATS);
    await this._writeJson(this._indexUri(), keep);
    for (const c of drop) {
      try {
        await vscode.workspace.fs.delete(this._chatUri(c.id));
      } catch {
        /* already gone */
      }
    }
  }

  // ── chats ──────────────────────────────────────────────────────────────────

  newChatId(): string {
    return `${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;
  }

  async loadChat(id: string): Promise<StoredChat | null> {
    const chat = await this._readJson<StoredChat | null>(this._chatUri(id), null);
    return chat;
  }

  async saveChat(id: string, title: string, snapshot: ChatSnapshot): Promise<ChatMeta> {
    const existing = await this.loadChat(id);
    const now = Date.now();
    const meta: ChatMeta = {
      id,
      title: title || existing?.title || 'New chat',
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
      messageCount: snapshot.uiMessages.length,
    };
    const stored: StoredChat = { ...meta, ...snapshot };
    await this._writeJson(this._chatUri(id), stored);
    await this._upsertIndex(meta);
    await this._pruneToLimit();
    return meta;
  }

  async deleteChat(id: string): Promise<void> {
    await this._removeFromIndex(id);
    try {
      await vscode.workspace.fs.delete(this._chatUri(id));
    } catch {
      /* file may not exist yet */
    }
    if (this.getActiveChatId() === id) {
      await this.setActiveChatId(null);
    }
  }

  async renameChat(id: string, title: string): Promise<ChatMeta | null> {
    const chat = await this.loadChat(id);
    if (!chat) return null;
    // Build meta from explicit ChatMeta fields only — spreading the full
    // StoredChat would write uiMessages/llmHistory into the lightweight index.
    const meta: ChatMeta = {
      id: chat.id,
      title,
      createdAt: chat.createdAt,
      updatedAt: chat.updatedAt,
      messageCount: chat.messageCount,
    };
    const stored: StoredChat = { ...chat, title };
    await this._writeJson(this._chatUri(id), stored);
    await this._upsertIndex(meta);
    return meta;
  }
}
