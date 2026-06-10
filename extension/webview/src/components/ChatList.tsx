import { useEffect, useRef } from 'react';
import type { ChatMeta } from '../types';

interface ChatListProps {
  chats: ChatMeta[];
  activeChatId: string | null;
  onSelect: (chatId: string) => void;
  onDelete: (chatId: string) => void;
  onClose: () => void;
}

function relativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const min = Math.floor(diff / 60000);
  if (min < 1) return 'just now';
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  return `${day}d ago`;
}

export function ChatList({ chats, activeChatId, onSelect, onDelete, onClose }: ChatListProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    }
    function onEsc(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onEsc);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onEsc);
    };
  }, [onClose]);

  return (
    <div className="chat-list" ref={ref}>
      <div className="chat-list__header">Chats</div>
      {chats.length === 0 ? (
        <div className="chat-list__empty">No saved chats yet</div>
      ) : (
        <ul className="chat-list__items">
          {chats.map((c) => (
            <li
              key={c.id}
              className={`chat-list__item ${c.id === activeChatId ? 'chat-list__item--active' : ''}`}
              onClick={() => onSelect(c.id)}
            >
              <div className="chat-list__item-main">
                <span className="chat-list__item-title" title={c.title}>{c.title}</span>
                <span className="chat-list__item-meta">{relativeTime(c.updatedAt)}</span>
              </div>
              <button
                className="chat-list__item-delete"
                title="Delete chat"
                aria-label="Delete chat"
                onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
