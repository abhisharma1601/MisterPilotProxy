import { useState } from 'react';
import type { PendingTerminalData } from '../types';

interface Props {
  terminal: PendingTerminalData;
  onApprove: (id: string) => void;
  onDeny: (id: string) => void;
}

export function TerminalCard({ terminal, onApprove, onDeny }: Props) {
  const { terminalId, command, status, stdout, stderr, exitCode, durationMs, timedOut, errorMessage } = terminal;
  const [submitted, setSubmitted] = useState(false);

  const isSuccess = status === 'done' && exitCode === 0;
  const isFailure = status === 'done' && exitCode !== 0;

  return (
    <div className={`terminal-card terminal-card--${status}`}>
      <div className="terminal-card__header">
        <span className="terminal-card__prompt">$</span>
        <code className="terminal-card__command">{command}</code>
        <span className="terminal-card__badge">
          {status === 'pending' && !submitted && '⚡'}
          {(status === 'pending' && submitted) && '⏳'}
          {isSuccess && '✓'}
          {isFailure && '✗'}
          {status === 'denied' && '🚫'}
          {status === 'error' && '⚠'}
        </span>
      </div>

      {status === 'pending' && !submitted && (
        <div className="terminal-card__approval">
          <button
            className="terminal-card__btn terminal-card__btn--allow"
            onClick={() => { setSubmitted(true); onApprove(terminalId); }}
          >
            Allow
          </button>
          <button
            className="terminal-card__btn terminal-card__btn--deny"
            onClick={() => { setSubmitted(true); onDeny(terminalId); }}
          >
            Deny
          </button>
        </div>
      )}

      {status === 'pending' && submitted && (
        <div className="terminal-card__info">Running…</div>
      )}

      {status === 'denied' && (
        <div className="terminal-card__info terminal-card__info--denied">
          Command denied
        </div>
      )}

      {status === 'error' && (
        <div className="terminal-card__info terminal-card__info--error">
          {errorMessage ?? 'An error occurred'}
        </div>
      )}

      {status === 'done' && (
        <div className="terminal-card__output">
          {stdout && <pre className="terminal-card__stdout">{stdout}</pre>}
          {stderr && <pre className="terminal-card__stderr">{stderr}</pre>}
          <div className="terminal-card__meta">
            {timedOut && <span className="terminal-card__timed-out">timed out · </span>}
            exit {exitCode} · {durationMs}ms
          </div>
        </div>
      )}
    </div>
  );
}
