import { useState, useCallback, useEffect, useRef } from 'react';
import { MessageList } from './MessageList';
import { InputBar } from './InputBar';
import { postToExtension } from '../vscode';
import type { ChatMessage, ExtensionMessage } from '../types';

let _idCounter = 0;
function uid(): string {
  return `${Date.now()}-${++_idCounter}`;
}

function isThinkingPlaceholder(prev: ChatMessage[]): boolean {
  const last = prev.at(-1);
  return !!(last?.isStreaming && !last.content && !last.toolCall && !last.pendingEdit && !last.pendingTerminal);
}

export function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [workspaceRoot, setWorkspaceRoot] = useState<string | null>(null);
  const [apiKeySet, setApiKeySet] = useState<boolean | null>(null);
  const [backendUrl, setBackendUrl] = useState<string | null>(null);

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

        case 'backendUrlStatus':
          setBackendUrl(msg.url);
          break;

        // chat streaming
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
  }, []);

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
      });
    },
    [streaming]
  );

  const handleStop = useCallback(() => {
    postToExtension({ type: 'stopChat' });
  }, []);

  const handleClear = useCallback(() => {
    setMessages([]);
    llmHistoryRef.current = [];
    pendingTurnRef.current = { content: '', toolCalls: [] };
    postToExtension({ type: 'clearHistory' });
  }, []);

  const handleSetApiKey = useCallback(() => {
    postToExtension({ type: 'setApiKey' });
  }, []);

  const handleSetBackendUrl = useCallback(() => {
    postToExtension({ type: 'setBackendUrl' });
  }, []);

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
          onClick={handleSetBackendUrl}
          title={backendUrl ? `Backend: ${backendUrl}` : 'Set backend URL'}
          aria-label="Set backend URL"
        >
          🌐
        </button>
        <button
          className="chat-panel__icon-btn"
          onClick={handleSetApiKey}
          title="Set API key"
          aria-label="Set API key"
        >
          🔑
        </button>
        <button className="chat-panel__icon-btn" onClick={handleClear} aria-label="Clear history">
          🗑
        </button>
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
      {!backendUrl && (
        <div className="api-key-banner">
          No backend URL configured.{' '}
          <button className="api-key-banner__btn" onClick={handleSetBackendUrl}>
            Set URL
          </button>
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
