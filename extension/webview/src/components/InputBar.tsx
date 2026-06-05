import React, { useState, useRef, useCallback } from 'react';

interface Props {
  onSend: (content: string) => void;
  onStop: () => void;
  disabled: boolean;
}

export function InputBar({ onSend, onStop, disabled }: Props) {
  const [value, setValue] = useState('');
  const ref = useRef<HTMLTextAreaElement>(null);

  const submit = useCallback(() => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue('');
    if (ref.current) ref.current.style.height = 'auto';
  }, [value, disabled, onSend]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submit();
      }
    },
    [submit]
  );

  const onChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  return (
    <div className="input-bar">
      <textarea
        ref={ref}
        className="input-textarea"
        value={value}
        onChange={onChange}
        onKeyDown={onKeyDown}
        placeholder="Ask anything… (Enter to send, Shift+Enter for newline)"
        disabled={disabled}
        rows={1}
      />
      {disabled ? (
        <button
          className="stop-button"
          onClick={onStop}
          title="Stop generation"
          aria-label="Stop generation"
        >
          ■
        </button>
      ) : (
        <button
          className="send-button"
          onClick={submit}
          disabled={!value.trim()}
          title="Send"
          aria-label="Send message"
        >
          ➤
        </button>
      )}
    </div>
  );
}
