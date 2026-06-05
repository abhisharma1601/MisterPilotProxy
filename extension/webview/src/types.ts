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

export type ExtensionMessage =
  | { type: 'chunk'; content: string }
  | { type: 'done' }
  | { type: 'error'; message: string }
  | { type: 'workspaceRoot'; path: string | null }
  | { type: 'apiKeyStatus'; isSet: boolean }
  | { type: 'backendUrlStatus'; url: string }
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
  | { type: 'toolResult'; id: string; tool: string; content: string };

export type WebviewMessage =
  | { type: 'chat'; messages: Record<string, unknown>[] }
  | { type: 'ready' }
  | { type: 'clearHistory' }
  | { type: 'stopChat' }
  | { type: 'setApiKey' }
  | { type: 'setBackendUrl' }
  | { type: 'applyEdit'; editId: string }
  | { type: 'rejectEdit'; editId: string }
  | { type: 'viewDiff'; path: string; original: string; proposed: string }
  | { type: 'approveTerminal'; id: string; approved: boolean };
