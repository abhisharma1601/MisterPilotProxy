export interface ToolCallData {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  result?: string;
  status: 'running' | 'done';
}

export interface PendingEditData {
  editId: string;
  path: string;
  diff: string;
  original: string;
  proposed: string;
  isNewFile: boolean;
  status: 'pending' | 'applied' | 'rejected';
}

export interface PendingTerminalData {
  terminalId: string;
  command: string;
  status: 'pending' | 'done' | 'denied' | 'error';
  stdout?: string;
  stderr?: string;
  exitCode?: number;
  durationMs?: number;
  timedOut?: boolean;
  errorMessage?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
  pendingEdit?: PendingEditData;
  pendingTerminal?: PendingTerminalData;
  toolCall?: ToolCallData;
}

export interface ChatMeta {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messageCount: number;
}

export interface ChatSnapshot {
  uiMessages: ChatMessage[];
  llmHistory: Record<string, unknown>[];
  model: string;
  mode: string;
  sessionCost: number;
}

export interface StoredChat extends ChatMeta, ChatSnapshot {}

export type ExtensionMessage =
  | { type: 'chunk'; content: string }
  | { type: 'sanitized_input'; content: string }
  | { type: 'done' }
  | { type: 'error'; message: string }
  | { type: 'cost'; usd: number }
  | { type: 'workspaceRoot'; path: string | null }
  | { type: 'apiKeyStatus'; isSet: boolean }
  // edit
  | { type: 'pendingEdit'; editId: string; path: string; diff: string; original: string; proposed: string; isNewFile: boolean }
  | { type: 'editApplied'; editId: string; path: string }
  | { type: 'editRejected'; editId: string; path: string }
  | { type: 'editError'; editId: string; message: string }
  // terminal
  | { type: 'terminalPending'; id: string; command: string }
  | { type: 'terminalResult'; id: string; command: string; approved: boolean; stdout: string; stderr: string; exitCode: number; durationMs: number; timedOut: boolean; error: string | null }
  | { type: 'terminalError'; id: string; command: string; message: string }
  // agent tool calls
  | { type: 'toolCall'; id: string; tool: string; args: Record<string, unknown> }
  | { type: 'toolResult'; id: string; tool: string; content: string }
  // chat persistence
  | { type: 'chatList'; chats: ChatMeta[]; activeChatId: string | null }
  | { type: 'chatLoaded'; chat: StoredChat }
  | { type: 'chatCreated'; chatId: string };

export type WebviewMessage =
  | { type: 'chat'; messages: Record<string, unknown>[]; model: string; mode: 'agent' | 'ask' }
  | { type: 'ready' }
  | { type: 'stopChat' }
  | { type: 'setApiKey' }
  | { type: 'applyEdit'; editId: string }
  | { type: 'rejectEdit'; editId: string }
  | { type: 'viewDiff'; path: string; original: string; proposed: string }
  | { type: 'approveTerminal'; id: string; approved: boolean }
  // chat persistence
  | { type: 'newChat' }
  | { type: 'saveChat'; chatId: string; title: string; snapshot: ChatSnapshot }
  | { type: 'loadChat'; chatId: string }
  | { type: 'listChats' }
  | { type: 'deleteChat'; chatId: string }
  | { type: 'renameChat'; chatId: string; title: string };
