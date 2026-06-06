export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export type WebviewMessage =
  | { type: 'chat'; messages: Record<string, unknown>[]; model: string; mode: 'agent' | 'ask' }
  | { type: 'ready' }
  | { type: 'clearHistory' }
  | { type: 'stopChat' }
  | { type: 'setApiKey' }
  | { type: 'setBackendUrl' }
  | { type: 'applyEdit'; editId: string }
  | { type: 'rejectEdit'; editId: string }
  | { type: 'viewDiff'; path: string; original: string; proposed: string }
  | { type: 'approveTerminal'; id: string; approved: boolean };

export type ExtensionMessage =
  | { type: 'chunk'; content: string }
  | { type: 'done' }
  | { type: 'error'; message: string }
  | { type: 'cost'; usd: number }
  | { type: 'workspaceRoot'; path: string | null }
  | { type: 'apiKeyStatus'; isSet: boolean }
  | { type: 'backendUrlStatus'; url: string }
  | { type: 'pendingEdit'; editId: string; path: string; diff: string; original: string; proposed: string; isNewFile: boolean }
  | { type: 'editApplied'; editId: string; path: string }
  | { type: 'editRejected'; editId: string; path: string }
  | { type: 'editError'; editId: string; message: string }
  | { type: 'terminalPending'; id: string; command: string }
  | { type: 'terminalResult'; id: string; command: string; approved: boolean; stdout: string; stderr: string; exitCode: number; durationMs: number; timedOut: boolean; error: string | null }
  | { type: 'terminalError'; id: string; command: string; message: string };
