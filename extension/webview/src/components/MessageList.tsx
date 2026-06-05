import { useEffect, useRef } from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { DiffCard } from './DiffCard';
import { TerminalCard } from './TerminalCard';
import type { ChatMessage, ToolCallData } from '../types';

marked.use({ breaks: true, gfm: true });

function md(raw: string): string {
  return DOMPurify.sanitize(marked.parse(raw) as string);
}

const TOOL_ICONS: Record<string, string> = {
  read_file: '📄',
  list_files: '📁',
  search_code: '🔍',
  write_file: '✏️',
  replace_in_file: '✏️',
  execute_terminal: '⚡',
};

function ToolCallCard({ toolCall }: { toolCall: ToolCallData }) {
  const icon = TOOL_ICONS[toolCall.tool] ?? '🔧';
  const cls = `tool-call-card tool-call-card--${toolCall.status}`;
  return (
    <div className={cls}>
      <span className="tool-call-card__icon">{icon}</span>
      <code className="tool-call-card__name">{toolCall.tool}</code>
      {toolCall.status === 'running' && (
        <span className="tool-call-card__spinner" aria-label="running" />
      )}
      {toolCall.status === 'done' && (
        <span className="tool-call-card__check">✓</span>
      )}
    </div>
  );
}

function ThinkingDots() {
  return (
    <div className="thinking-dots" aria-label="Thinking">
      <span /><span /><span />
    </div>
  );
}

interface ItemProps {
  msg: ChatMessage;
  onApply: (editId: string) => void;
  onReject: (editId: string) => void;
  onViewFull: (editId: string, path: string, original: string, proposed: string) => void;
  onApproveTerminal: (id: string) => void;
  onDenyTerminal: (id: string) => void;
}

function MessageItem({ msg, onApply, onReject, onViewFull, onApproveTerminal, onDenyTerminal }: ItemProps) {
  if (msg.toolCall) {
    return (
      <div className="message message--assistant">
        <ToolCallCard toolCall={msg.toolCall} />
      </div>
    );
  }

  if (msg.pendingTerminal) {
    return (
      <div className="message message--assistant">
        {msg.content && (
          <div
            className="message__content message__content--md"
            dangerouslySetInnerHTML={{ __html: md(msg.content) }}
          />
        )}
        <TerminalCard
          terminal={msg.pendingTerminal}
          onApprove={onApproveTerminal}
          onDeny={onDenyTerminal}
        />
      </div>
    );
  }

  if (msg.pendingEdit) {
    return (
      <div className="message message--assistant">
        {msg.content && (
          <div
            className="message__content message__content--md"
            dangerouslySetInnerHTML={{ __html: md(msg.content) }}
          />
        )}
        <DiffCard
          edit={msg.pendingEdit}
          onApply={onApply}
          onReject={onReject}
          onViewFull={onViewFull}
        />
      </div>
    );
  }

  if (msg.role === 'user') {
    return (
      <div className="message message--user">
        <div className="message__role">You</div>
        <div className="message__content">{msg.content}</div>
      </div>
    );
  }

  // Thinking placeholder: streaming + empty content
  if (msg.isStreaming && !msg.content) {
    return (
      <div className="message message--assistant">
        <div className="message__role">MisterPilot</div>
        <ThinkingDots />
      </div>
    );
  }

  return (
    <div className="message message--assistant">
      <div className="message__role">MisterPilot</div>
      <div
        className="message__content message__content--md"
        dangerouslySetInnerHTML={{ __html: md(msg.content) }}
      />
      {msg.isStreaming && <span className="message__cursor">▌</span>}
    </div>
  );
}

interface ListProps {
  messages: ChatMessage[];
  onApply: (editId: string) => void;
  onReject: (editId: string) => void;
  onViewFull: (editId: string, path: string, original: string, proposed: string) => void;
  onApproveTerminal: (id: string) => void;
  onDenyTerminal: (id: string) => void;
}

export function MessageList({ messages, onApply, onReject, onViewFull, onApproveTerminal, onDenyTerminal }: ListProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state__icon">🚀</div>
        <div className="empty-state__text">Ask MisterPilot anything about your codebase</div>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((m) => (
        <MessageItem
          key={m.id}
          msg={m}
          onApply={onApply}
          onReject={onReject}
          onViewFull={onViewFull}
          onApproveTerminal={onApproveTerminal}
          onDenyTerminal={onDenyTerminal}
        />
      ))}
      <div ref={endRef} />
    </div>
  );
}
