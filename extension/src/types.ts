export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatSnapshot {
  uiMessages: unknown[];
  llmHistory: Record<string, unknown>[];
  model: string;
  mode: string;
  sessionCost: number;
  sessionCostInr: number;
}

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

export type ExtensionMessage =
  | { type: 'chunk'; content: string }
  | { type: 'sanitized_input'; content: string }
  | { type: 'done' }
  | { type: 'error'; message: string }
  | { type: 'cost'; usd: number; inr?: number }
  | { type: 'workspaceRoot'; path: string | null }
  | { type: 'apiKeyStatus'; isSet: boolean }
  | { type: 'pendingEdit'; editId: string; path: string; diff: string; original: string; proposed: string; isNewFile: boolean }
  | { type: 'editApplied'; editId: string; path: string }
  | { type: 'editRejected'; editId: string; path: string }
  | { type: 'editError'; editId: string; message: string }
  | { type: 'terminalPending'; id: string; command: string }
  | { type: 'terminalResult'; id: string; command: string; approved: boolean; stdout: string; stderr: string; exitCode: number; durationMs: number; timedOut: boolean; error: string | null }
  | { type: 'terminalError'; id: string; command: string; message: string }
  // chat persistence
  | { type: 'chatList'; chats: ChatMetaDTO[]; activeChatId: string | null }
  | { type: 'chatLoaded'; chat: StoredChatDTO }
  | { type: 'chatCreated'; chatId: string }
  | { type: 'models'; models: string[] };

export interface ChatMetaDTO {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messageCount: number;
}

export interface StoredChatDTO extends ChatMetaDTO, ChatSnapshot {}
