import { useState, useEffect } from 'react';
import { postToExtension } from '../vscode';
import type { McpServerStatus } from '../types';

interface McpServerConfig {
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  type?: 'http';
  url?: string;
  headers?: Record<string, string>;
}

interface McpPageProps {
  servers: McpServerStatus[];
  onClose: () => void;
}

export function McpPage({ servers, onClose }: McpPageProps) {
  const [config, setConfig] = useState<Record<string, McpServerConfig>>({});
  const [showAddForm, setShowAddForm] = useState(false);

  const [name, setName] = useState('');
  const [serverType, setServerType] = useState<'stdio' | 'http'>('stdio');
  const [command, setCommand] = useState('');
  const [args, setArgs] = useState('');
  const [envPairs, setEnvPairs] = useState<[string, string][]>([['', '']]);
  const [url, setUrl] = useState('');
  const [headerPairs, setHeaderPairs] = useState<[string, string][]>([['', '']]);

  useEffect(() => {
    postToExtension({ type: 'getMcpConfig' });
    function onMsg(e: MessageEvent) {
      if (e.data?.type === 'mcpConfig') setConfig(e.data.config ?? {});
    }
    window.addEventListener('message', onMsg);
    return () => window.removeEventListener('message', onMsg);
  }, []);

  function pairsToRecord(pairs: [string, string][]): Record<string, string> {
    const out: Record<string, string> = {};
    for (const [k, v] of pairs) if (k.trim()) out[k.trim()] = v;
    return out;
  }

  function saveConfig(next: Record<string, McpServerConfig>) {
    setConfig(next);
    postToExtension({ type: 'saveMcpConfig', config: next });
  }

  function handleAdd() {
    if (!name.trim()) return;
    const next = { ...config };
    if (serverType === 'http') {
      const headers = pairsToRecord(headerPairs);
      next[name.trim()] = {
        type: 'http',
        url: url.trim(),
        ...(Object.keys(headers).length > 0 ? { headers } : {}),
      };
    } else {
      const env = pairsToRecord(envPairs);
      const argsArr = args.trim() ? args.trim().split(/\s+/) : [];
      next[name.trim()] = {
        command: command.trim(),
        ...(argsArr.length > 0 ? { args: argsArr } : {}),
        ...(Object.keys(env).length > 0 ? { env } : {}),
      };
    }
    saveConfig(next);
    setName(''); setServerType('stdio'); setCommand(''); setArgs('');
    setEnvPairs([['', '']]); setUrl(''); setHeaderPairs([['', '']]);
    setShowAddForm(false);
  }

  function handleRemove(sName: string) {
    const next = { ...config };
    delete next[sName];
    saveConfig(next);
  }

  function updateEnv(i: number, k: string, v: string) {
    const next = [...envPairs] as [string, string][];
    next[i] = [k, v];
    setEnvPairs(next);
  }

  function updateHeader(i: number, k: string, v: string) {
    const next = [...headerPairs] as [string, string][];
    next[i] = [k, v];
    setHeaderPairs(next);
  }

  const allNames = Array.from(new Set([...Object.keys(config), ...servers.map(s => s.name)]));

  return (
    <div className="mcp-page">
      <div className="mcp-page__header">
        <span className="mcp-page__title">MCP Servers</span>
        <button className="mcp-page__header-btn" onClick={() => setShowAddForm(v => !v)}>
          {showAddForm ? 'Cancel' : '+ Add'}
        </button>
        <button
          className="mcp-page__header-btn mcp-page__header-btn--ghost"
          onClick={() => postToExtension({ type: 'openMcpSettings' })}
          title="Edit config JSON directly in VS Code settings"
        >
          Edit JSON
        </button>
        <button className="mcp-page__close" onClick={onClose} title="Close">✕</button>
      </div>

      <div className="mcp-page__body">
        {allNames.length === 0 && !showAddForm && (
          <div className="mcp-page__empty">No servers configured. Click + Add to get started.</div>
        )}

        {allNames.map(sName => {
          const cfg = config[sName];
          const status = servers.find(s => s.name === sName);
          const dot = status?.connected === null || status?.connected === undefined
            ? 'checking' : status.connected ? 'ok' : 'err';
          const isHttp = cfg?.type === 'http';
          const preview = isHttp
            ? cfg?.url ?? ''
            : [cfg?.command, ...(cfg?.args ?? [])].filter(Boolean).join(' ');

          return (
            <div key={sName} className="mcp-server-row">
              <div className="mcp-server-row__main">
                <span className={`mcp-server-row__dot mcp-server-row__dot--${dot}`}>●</span>
                <span className="mcp-server-row__name">{sName}</span>
                <span className={`mcp-server-row__badge mcp-server-row__badge--${isHttp ? 'http' : 'stdio'}`}>
                  {isHttp ? 'HTTP' : 'STDIO'}
                </span>
                {status?.connected && status.toolCount !== undefined && (
                  <span className="mcp-server-row__tools">{status.toolCount} tools</span>
                )}
                {status?.connected === false && (
                  <span className="mcp-server-row__failed">failed</span>
                )}
                <button
                  className="mcp-server-row__remove"
                  onClick={() => handleRemove(sName)}
                  title="Remove"
                >✕</button>
              </div>
              {preview && <div className="mcp-server-row__preview">{preview}</div>}
            </div>
          );
        })}

        {showAddForm && (
          <div className="mcp-form">
            <div className="mcp-form__row">
              <label className="mcp-form__label">Name</label>
              <input className="mcp-form__input" value={name} onChange={e => setName(e.target.value)} placeholder="github" />
            </div>

            <div className="mcp-form__row">
              <label className="mcp-form__label">Type</label>
              <div className="mcp-form__toggle">
                <button
                  className={`mcp-form__toggle-btn${serverType === 'stdio' ? ' mcp-form__toggle-btn--active' : ''}`}
                  onClick={() => setServerType('stdio')}
                >STDIO</button>
                <button
                  className={`mcp-form__toggle-btn${serverType === 'http' ? ' mcp-form__toggle-btn--active' : ''}`}
                  onClick={() => setServerType('http')}
                >HTTP</button>
              </div>
            </div>

            {serverType === 'stdio' ? (
              <>
                <div className="mcp-form__row">
                  <label className="mcp-form__label">Command</label>
                  <input className="mcp-form__input" value={command} onChange={e => setCommand(e.target.value)} placeholder="npx" />
                </div>
                <div className="mcp-form__row">
                  <label className="mcp-form__label">Args</label>
                  <input className="mcp-form__input" value={args} onChange={e => setArgs(e.target.value)} placeholder="-y @modelcontextprotocol/server-github" />
                </div>
                <div className="mcp-form__section">
                  <label className="mcp-form__label">Env vars</label>
                  {envPairs.map(([k, v], i) => (
                    <div key={i} className="mcp-form__kv">
                      <input className="mcp-form__input mcp-form__input--kv" value={k} onChange={e => updateEnv(i, e.target.value, v)} placeholder="KEY" />
                      <input className="mcp-form__input mcp-form__input--kv" value={v} onChange={e => updateEnv(i, k, e.target.value)} placeholder="value" type="password" />
                      <button className="mcp-form__kv-del" onClick={() => setEnvPairs(envPairs.filter((_, j) => j !== i))}>✕</button>
                    </div>
                  ))}
                  <button className="mcp-form__add-pair" onClick={() => setEnvPairs([...envPairs, ['', '']])}>+ Add var</button>
                </div>
              </>
            ) : (
              <>
                <div className="mcp-form__row">
                  <label className="mcp-form__label">URL</label>
                  <input className="mcp-form__input" value={url} onChange={e => setUrl(e.target.value)} placeholder="https://api.githubcopilot.com/mcp/" />
                </div>
                <div className="mcp-form__section">
                  <label className="mcp-form__label">Headers</label>
                  {headerPairs.map(([k, v], i) => (
                    <div key={i} className="mcp-form__kv">
                      <input className="mcp-form__input mcp-form__input--kv" value={k} onChange={e => updateHeader(i, e.target.value, v)} placeholder="Authorization" />
                      <input className="mcp-form__input mcp-form__input--kv" value={v} onChange={e => updateHeader(i, k, e.target.value)} placeholder="Bearer ..." type="password" />
                      <button className="mcp-form__kv-del" onClick={() => setHeaderPairs(headerPairs.filter((_, j) => j !== i))}>✕</button>
                    </div>
                  ))}
                  <button className="mcp-form__add-pair" onClick={() => setHeaderPairs([...headerPairs, ['', '']])}>+ Add header</button>
                </div>
              </>
            )}

            <button
              className="mcp-form__submit"
              onClick={handleAdd}
              disabled={!name.trim() || (serverType === 'stdio' ? !command.trim() : !url.trim())}
            >
              Add Server
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
