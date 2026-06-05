import type { PendingEditData } from '../types';

// ── Diff line renderer ────────────────────────────────────────────────────────

function DiffViewer({ diff }: { diff: string }) {
  if (!diff.trim()) {
    return <div className="diff-viewer diff-viewer--empty">(no diff available)</div>;
  }

  const lines = diff.split('\n');

  return (
    <pre className="diff-viewer">
      {lines.map((line, i) => {
        let cls = 'diff-line';
        if (line.startsWith('+++') || line.startsWith('---')) {
          cls += ' diff-line--header';
        } else if (line.startsWith('@@')) {
          cls += ' diff-line--hunk';
        } else if (line.startsWith('+')) {
          cls += ' diff-line--add';
        } else if (line.startsWith('-')) {
          cls += ' diff-line--remove';
        } else {
          cls += ' diff-line--context';
        }
        return (
          <div key={i} className={cls}>
            {line || ' '}
          </div>
        );
      })}
    </pre>
  );
}

// ── DiffCard ──────────────────────────────────────────────────────────────────

interface DiffCardProps {
  edit: PendingEditData;
  onApply: (editId: string) => void;
  onReject: (editId: string) => void;
  onViewFull: (editId: string, path: string, original: string, proposed: string) => void;
}

export function DiffCard({ edit, onApply, onReject, onViewFull }: DiffCardProps) {
  const { editId, path, diff, original, proposed, isNewFile, status } = edit;

  if (status === 'applied') {
    return (
      <div className="diff-card diff-card--resolved diff-card--applied">
        <span className="diff-card__icon">✅</span>
        <span>Applied changes to <code>{path}</code></span>
      </div>
    );
  }

  if (status === 'rejected') {
    return (
      <div className="diff-card diff-card--resolved diff-card--rejected">
        <span className="diff-card__icon">✗</span>
        <span>Rejected edit to <code>{path}</code></span>
      </div>
    );
  }

  return (
    <div className="diff-card">
      <div className="diff-card__header">
        <span className="diff-card__file-icon">{isNewFile ? '🆕' : '📝'}</span>
        <span className="diff-card__path" title={path}>{path}</span>
        <span className="diff-card__badge">{isNewFile ? 'new file' : 'edit'}</span>
      </div>

      <DiffViewer diff={diff} />

      <div className="diff-card__actions">
        <button
          className="diff-card__btn diff-card__btn--apply"
          onClick={() => onApply(editId)}
          title="Write this change to disk"
        >
          ✓ Apply
        </button>
        <button
          className="diff-card__btn diff-card__btn--reject"
          onClick={() => onReject(editId)}
          title="Discard this proposed change"
        >
          ✗ Reject
        </button>
        <button
          className="diff-card__btn diff-card__btn--view"
          onClick={() => onViewFull(editId, path, original, proposed)}
          title="Open in VS Code diff editor"
        >
          ⊞ Full diff
        </button>
      </div>
    </div>
  );
}
