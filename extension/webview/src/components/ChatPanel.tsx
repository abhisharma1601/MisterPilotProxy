import { useState, useCallback, useEffect, useRef } from 'react';
import { MessageList } from './MessageList';
import { InputBar } from './InputBar';
import { ChatList } from './ChatList';
import { postToExtension } from '../vscode';
import type { ChatMessage, ChatMeta, ExtensionMessage } from '../types';

let _idCounter = 0;
function uid(): string {
  return `${Date.now()}-${++_idCounter}`;
}

function isThinkingPlaceholder(prev: ChatMessage[]): boolean {
  const last = prev.at(-1);
  return !!(last?.isStreaming && !last.content && !last.toolCall && !last.pendingEdit && !last.pendingTerminal);
}

// A chat title derived from its first user message.
function deriveTitle(msgs: ChatMessage[]): string {
  const firstUser = msgs.find((m) => m.role === 'user' && m.content.trim());
  if (!firstUser) return 'New chat';
  const t = firstUser.content.trim().replace(/\s+/g, ' ');
  return t.length > 48 ? `${t.slice(0, 48)}…` : t;
}

// Pending edits/terminals can't be resumed after a reload (their local op state
// lives in the extension host and is gone), so mark restored ones as resolved.
function sanitizeLoaded(msgs: ChatMessage[]): ChatMessage[] {
  return msgs.map((m) => {
    let next = m;
    if (m.pendingEdit?.status === 'pending') {
      next = { ...next, pendingEdit: { ...m.pendingEdit, status: 'rejected' } };
    }
    if (m.pendingTerminal?.status === 'pending') {
      next = { ...next, pendingTerminal: { ...m.pendingTerminal, status: 'denied' } };
    }
    return next.isStreaming ? { ...next, isStreaming: false } : next;
  });
}

export function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [workspaceRoot, setWorkspaceRoot] = useState<string | null>(null);
  const [apiKeySet, setApiKeySet] = useState<boolean | null>(null);
  const [model, setModel] = useState('deepseek-v4-pro');
  const [mode, setMode] = useState<'agent' | 'ask'>('agent');
  const [sessionCost, setSessionCost] = useState(0);
  const [chats, setChats] = useState<ChatMeta[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [showChatList, setShowChatList] = useState(false);
  const INR_RATE = 96;

  // Ref mirror of the active chat id so the (mount-only) message handler and the
  // persistence effect always read the current value.
  const activeChatIdRef = useRef<string | null>(null);
  const setActive = useCallback((id: string | null) => {
    activeChatIdRef.current = id;
    setActiveChatId(id);
  }, []);

  // Set when a chat is loaded/restored so the immediately-following save effect
  // is skipped — viewing a chat shouldn't bump its updatedAt and reorder the list.
  const skipNextSaveRef = useRef(false);
  // True when there are unsaved changes pending the debounce; lets us flush on hide.
  const dirtyRef = useRef(false);

  // Tracks the full LLM conversation in OpenAI format for multi-turn tool use
  const llmHistoryRef = useRef<Record<string, unknown>[]>([]);
  // Accumulates content/toolCalls for the current in-flight assistant turn
  const pendingTurnRef = useRef<{
    content: string;
    toolCalls: Array<{ id: string; tool: string; args: Record<string, unknown> }>;
  }>({ content: '', toolCalls: [] });

  // ── extension message handler ───────────────────────────────────────────

  useEffect(() => {
    postToExtension({ type: 'ready' });

    function onMessage(event: MessageEvent<ExtensionMessage>) {
      const msg = event.data;

      switch (msg.type) {
        // workspace
        case 'workspaceRoot':
          setWorkspaceRoot(msg.path);
          break;

        case 'apiKeyStatus':
          setApiKeySet(msg.isSet);
          break;

        // chat persistence
        case 'chatList':
          setChats(msg.chats);
          if (!activeChatIdRef.current && msg.activeChatId) setActive(msg.activeChatId);
          break;

        case 'chatCreated':
          setActive(msg.chatId);
          setMessages([]);
          llmHistoryRef.current = [];
          pendingTurnRef.current = { content: '', toolCalls: [] };
          setSessionCost(0);
          setShowChatList(false);
          break;

        case 'chatLoaded':
          skipNextSaveRef.current = true;
          dirtyRef.current = false;
          setActive(msg.chat.id);
          setMessages(sanitizeLoaded(msg.chat.uiMessages as ChatMessage[]));
          llmHistoryRef.current = msg.chat.llmHistory ?? [];
          pendingTurnRef.current = { content: '', toolCalls: [] };
          setModel(msg.chat.model || 'deepseek-v4-pro');
          setMode(msg.chat.mode === 'ask' ? 'ask' : 'agent');
          setSessionCost(msg.chat.sessionCost ?? 0);
          setStreaming(false);
          setShowChatList(false);
          break;

        // chat streaming
        case 'sanitized_input':
          setMessages((prev) => {
            const idx = [...prev].reverse().findIndex((m) => m.role === 'user');
            if (idx === -1) return prev;
            const realIdx = prev.length - 1 - idx;
            const updated = [...prev];
            updated[realIdx] = { ...updated[realIdx], content: msg.content };
            return updated;
          });
          break;

        case 'chunk':
          pendingTurnRef.current.content += msg.content;
          setMessages((prev) => {
            const last = prev.at(-1);
            if (last?.role === 'assistant' && last.isStreaming) {
              return [...prev.slice(0, -1), { ...last, content: last.content + msg.content }];
            }
            return [...prev, { id: uid(), role: 'assistant', content: msg.content, isStreaming: true }];
          });
          break;

        case 'done':
          // Commit any remaining prose as the final assistant message in LLM history
          if (pendingTurnRef.current.content) {
            llmHistoryRef.current = [
              ...llmHistoryRef.current,
              { role: 'assistant', content: pendingTurnRef.current.content },
            ];
          }
          pendingTurnRef.current = { content: '', toolCalls: [] };
          setStreaming(false);
          setMessages((prev) => {
            const last = prev.at(-1);
            if (!last?.isStreaming) return prev;
            if (!last.content && !last.toolCall && !last.pendingEdit && !last.pendingTerminal) {
              return prev.slice(0, -1);
            }
            return [...prev.slice(0, -1), { ...last, isStreaming: false }];
          });
          break;

        case 'error':
          pendingTurnRef.current = { content: '', toolCalls: [] };
          setStreaming(false);
          setMessages((prev) => {
            let updated = prev;
            const last = prev.at(-1);
            if (last?.isStreaming) {
              updated = [...prev.slice(0, -1), { ...last, isStreaming: false }];
            }
            return [...updated, { id: uid(), role: 'assistant', content: `**Error:** ${msg.message}` }];
          });
          break;

        case 'cost':
          setSessionCost((prev) => prev + msg.usd);
          break;

        // file edits
        case 'pendingEdit':
          setMessages((prev) => {
            const base = isThinkingPlaceholder(prev) ? prev.slice(0, -1) : prev;
            return [
              ...base,
              {
                id: uid(),
                role: 'assistant' as const,
                content: '',
                pendingEdit: {
                  editId: msg.editId,
                  path: msg.path,
                  diff: msg.diff,
                  original: msg.original,
                  proposed: msg.proposed,
                  isNewFile: msg.isNewFile,
                  status: 'pending' as const,
                },
              },
            ];
          });
          break;

        case 'editApplied':
          setMessages((prev) =>
            prev.map((m) =>
              m.pendingEdit?.editId === msg.editId
                ? { ...m, pendingEdit: { ...m.pendingEdit!, status: 'applied' } }
                : m
            )
          );
          break;

        case 'editRejected':
          setMessages((prev) =>
            prev.map((m) =>
              m.pendingEdit?.editId === msg.editId
                ? { ...m, pendingEdit: { ...m.pendingEdit!, status: 'rejected' } }
                : m
            )
          );
          break;

        case 'editError':
          setMessages((prev) =>
            prev.map((m) =>
              m.pendingEdit?.editId === msg.editId
                ? { ...m, pendingEdit: { ...m.pendingEdit!, status: 'rejected' }, content: `**Error:** ${msg.message}` }
                : m
            )
          );
          break;

        // terminal
        case 'terminalPending':
          setMessages((prev) => {
            const base = isThinkingPlaceholder(prev) ? prev.slice(0, -1) : prev;
            return [
              ...base,
              {
                id: uid(),
                role: 'assistant' as const,
                content: '',
                pendingTerminal: {
                  terminalId: msg.id,
                  command: msg.command,
                  status: 'pending' as const,
                },
              },
            ];
          });
          break;

        case 'terminalResult':
          setMessages((prev) =>
            prev.map((m) => {
              if (m.pendingTerminal?.terminalId !== msg.id) return m;
              return {
                ...m,
                pendingTerminal: {
                  ...m.pendingTerminal,
                  status: msg.approved ? 'done' : 'denied',
                  stdout: msg.stdout,
                  stderr: msg.stderr,
                  exitCode: msg.exitCode,
                  durationMs: msg.durationMs,
                  timedOut: msg.timedOut,
                  errorMessage: msg.error ?? undefined,
                },
              };
            })
          );
          break;

        case 'terminalError':
          setMessages((prev) =>
            prev.map((m) =>
              m.pendingTerminal?.terminalId === msg.id
                ? { ...m, pendingTerminal: { ...m.pendingTerminal!, status: 'error', errorMessage: msg.message } }
                : m
            )
          );
          break;

        // agent tool calls
        case 'toolCall':
          pendingTurnRef.current.toolCalls.push({ id: msg.id, tool: msg.tool, args: msg.args });
          setMessages((prev) => {
            const base = isThinkingPlaceholder(prev) ? prev.slice(0, -1) : prev;
            return [
              ...base,
              {
                id: uid(),
                role: 'assistant' as const,
                content: '',
                toolCall: {
                  id: msg.id,
                  tool: msg.tool,
                  args: msg.args,
                  status: 'running' as const,
                },
              },
            ];
          });
          break;

        case 'toolResult': {
          // Commit the assistant turn for this tool call into LLM history
          const tc = pendingTurnRef.current.toolCalls.find((t) => t.id === msg.id);
          if (tc) {
            llmHistoryRef.current = [
              ...llmHistoryRef.current,
              {
                role: 'assistant',
                content: pendingTurnRef.current.content || null,
                tool_calls: [
                  {
                    id: tc.id,
                    type: 'function',
                    function: { name: tc.tool, arguments: JSON.stringify(tc.args) },
                  },
                ],
              },
              {
                role: 'tool',
                tool_call_id: tc.id,
                content: msg.content,
              },
            ];
            // Content was committed with this tool call; clear it
            pendingTurnRef.current.content = '';
            pendingTurnRef.current.toolCalls = pendingTurnRef.current.toolCalls.filter(
              (t) => t.id !== msg.id
            );
          }
          setMessages((prev) =>
            prev.map((m) =>
              m.toolCall?.id === msg.id
                ? { ...m, toolCall: { ...m.toolCall!, result: msg.content, status: 'done' } }
                : m
            )
          );
          break;
        }
      }
    }

    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [setActive]);

  // ── persistence ───────────────────────────────────────────────────────────

  const postSave = useCallback(
    (chatId: string) => {
      postToExtension({
        type: 'saveChat',
        chatId,
        title: deriveTitle(messages),
        snapshot: {
          uiMessages: messages,
          llmHistory: llmHistoryRef.current,
          model,
          mode,
          sessionCost,
        },
      });
      dirtyRef.current = false;
    },
    [messages, model, mode, sessionCost]
  );

  // Debounced save at turn boundaries (never mid-stream). This also captures
  // edit/terminal status changes, which land after streaming ends.
  useEffect(() => {
    if (streaming) return;
    const chatId = activeChatIdRef.current;
    if (!chatId || messages.length === 0) return;
    // A load/restore just set this state — don't re-save (would bump updatedAt).
    if (skipNextSaveRef.current) { skipNextSaveRef.current = false; return; }
    dirtyRef.current = true;
    const handle = setTimeout(() => postSave(chatId), 600);
    return () => clearTimeout(handle);
  }, [messages, streaming, model, mode, sessionCost, postSave]);

  // Flush any pending unsaved change when the panel is hidden/closed, so the
  // last turn isn't lost to the debounce window on reload.
  useEffect(() => {
    function onHide() {
      if (document.visibilityState !== 'hidden') return;
      if (!dirtyRef.current) return;
      const chatId = activeChatIdRef.current;
      if (!chatId || messages.length === 0) return;
      postSave(chatId);
    }
    document.addEventListener('visibilitychange', onHide);
    return () => document.removeEventListener('visibilitychange', onHide);
  }, [messages, postSave]);

  // Close the history dropdown if a stream begins while it's open (switching is
  // blocked mid-stream anyway).
  useEffect(() => {
    if (streaming) setShowChatList(false);
  }, [streaming]);

  // ── chat send ───────────────────────────────────────────────────────────

  const handleSend = useCallback(
    (content: string) => {
      if (streaming) return;
      const userMsg: ChatMessage = { id: uid(), role: 'user', content };
      const thinkingMsg: ChatMessage = { id: uid(), role: 'assistant', content: '', isStreaming: true };
      setMessages((prev) => [...prev, userMsg, thinkingMsg]);
      setStreaming(true);

      // Add user message to LLM history and reset pending turn
      llmHistoryRef.current = [...llmHistoryRef.current, { role: 'user', content }];
      pendingTurnRef.current = { content: '', toolCalls: [] };

      postToExtension({
        type: 'chat',
        messages: llmHistoryRef.current,
        model,
        mode,
      });
    },
    [streaming, model, mode]
  );

  const handleStop = useCallback(() => {
    postToExtension({ type: 'stopChat' });
  }, []);

  const handleSetApiKey = useCallback(() => {
    postToExtension({ type: 'setApiKey' });
  }, []);

  // ── chat management ───────────────────────────────────────────────────────

  const handleNewChat = useCallback(() => {
    if (streaming) return;
    postToExtension({ type: 'newChat' });
  }, [streaming]);

  const handleToggleChatList = useCallback(() => {
    setShowChatList((prev) => {
      if (!prev) postToExtension({ type: 'listChats' });
      return !prev;
    });
  }, []);

  // Switching/deleting while a response streams would let the in-flight stream
  // bleed into the newly-loaded chat, so block it until the turn finishes.
  const handleSelectChat = useCallback((chatId: string) => {
    if (streaming) return;
    if (chatId === activeChatIdRef.current) { setShowChatList(false); return; }
    postToExtension({ type: 'loadChat', chatId });
  }, [streaming]);

  const handleDeleteChat = useCallback((chatId: string) => {
    if (streaming) return;
    postToExtension({ type: 'deleteChat', chatId });
  }, [streaming]);

  // ── edit actions ────────────────────────────────────────────────────────

  const handleApply = useCallback((editId: string) => {
    postToExtension({ type: 'applyEdit', editId });
  }, []);

  const handleReject = useCallback((editId: string) => {
    postToExtension({ type: 'rejectEdit', editId });
  }, []);

  const handleViewFull = useCallback(
    (_editId: string, path: string, original: string, proposed: string) => {
      postToExtension({ type: 'viewDiff', path, original, proposed });
    },
    []
  );

  // ── terminal approval ───────────────────────────────────────────────────

  const handleApproveTerminal = useCallback((id: string) => {
    postToExtension({ type: 'approveTerminal', id, approved: true });
  }, []);

  const handleDenyTerminal = useCallback((id: string) => {
    postToExtension({ type: 'approveTerminal', id, approved: false });
  }, []);

  const folderName = workspaceRoot
    ? (workspaceRoot.split('/').pop() ?? workspaceRoot)
    : null;

  return (
    <div className="chat-panel">
      <div className="chat-panel__header">
        <span className="chat-panel__title">MisterPilot</span>
        {folderName && (
          <span className="chat-panel__workspace" title={workspaceRoot ?? ''}>
            📁 {folderName}
          </span>
        )}
        <button
          className="chat-panel__icon-btn"
          onClick={handleNewChat}
          disabled={streaming}
          title="New chat"
          aria-label="New chat"
        >
          ＋
        </button>
        <div className="chat-panel__history-wrap">
          <button
            className="chat-panel__icon-btn"
            onClick={handleToggleChatList}
            disabled={streaming}
            title="Chat history"
            aria-label="Chat history"
          >
            🕘
          </button>
          {showChatList && (
            <ChatList
              chats={chats}
              activeChatId={activeChatId}
              onSelect={handleSelectChat}
              onDelete={handleDeleteChat}
              onClose={() => setShowChatList(false)}
            />
          )}
        </div>
        <button
          className="chat-panel__icon-btn"
          onClick={handleSetApiKey}
          title="Set API key"
          aria-label="Set API key"
        >
          🔑
        </button>
      </div>

      <div className="chat-panel__toolbar">
        <select
          className="model-select"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={streaming}
          title="Select model"
        >
          <option value="deepseek-v4-pro">deepseek-v4-pro</option>
          <option value="deepseek-v4-flash">deepseek-v4-flash</option>
        </select>
        <div className="mode-toggle">
          <button
            className={`mode-toggle__btn ${mode === 'agent' ? 'mode-toggle__btn--active' : ''}`}
            onClick={() => setMode('agent')}
            disabled={streaming}
            title="Agent mode: can read, write files and run commands"
          >
            Agent
          </button>
          <button
            className={`mode-toggle__btn ${mode === 'ask' ? 'mode-toggle__btn--active' : ''}`}
            onClick={() => setMode('ask')}
            disabled={streaming}
            title="Ask mode: read-only, no file changes or commands"
          >
            Ask
          </button>
        </div>
        {sessionCost > 0 && (
          <span className="session-cost" title="Session cost (resets on clear)">
            ${sessionCost.toFixed(4)} / ₹{(sessionCost * INR_RATE).toFixed(2)}
          </span>
        )}
      </div>

      {apiKeySet === false && (
        <div className="api-key-banner">
          No API key set.{' '}
          <button className="api-key-banner__btn" onClick={handleSetApiKey}>
            Set API key
          </button>
          {' '}or run <code>MisterPilot: Set API Key</code> from the command palette.
        </div>
      )}
      <div className="chat-panel__messages">
        <MessageList
          messages={messages}
          onApply={handleApply}
          onReject={handleReject}
          onViewFull={handleViewFull}
          onApproveTerminal={handleApproveTerminal}
          onDenyTerminal={handleDenyTerminal}
        />
      </div>

      <div className="chat-panel__input">
        <InputBar onSend={handleSend} onStop={handleStop} disabled={streaming} />
      </div>
    </div>
  );
}
