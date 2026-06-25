"""
MCP (Model Context Protocol) client manager.

Maintains long-lived stdio connections to configured MCP servers, discovers
their tools, and executes tool calls on behalf of the agent loop.

Connection lifecycle:
  - Sessions are created lazily on first use and cached by (name, config) key.
  - A background asyncio task holds the process alive inside the MCP context
    managers so the session stays open across multiple chat turns.
  - Dead sessions are evicted and reconnected transparently on next use.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_MCP_PREFIX = "mcp__"


def _make_tool_name(server_name: str, tool_name: str) -> str:
    return f"{_MCP_PREFIX}{server_name}__{tool_name}"


def _parse_tool_name(qualified: str) -> Optional[Tuple[str, str]]:
    if not qualified.startswith(_MCP_PREFIX):
        return None
    rest = qualified[len(_MCP_PREFIX):]
    parts = rest.split("__", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else None


class _ServerSession:
    """
    Background asyncio task that holds one live MCP stdio connection.

    The task blocks inside the MCP async context managers — keeping the
    subprocess alive — while external coroutines call list_tools / call_tool
    directly on the session object (safe within a single event loop).
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self._config = config
        self._session = None
        self._ready: asyncio.Event = asyncio.Event()
        self._shutdown: asyncio.Event = asyncio.Event()
        self._start_error: Optional[Exception] = None
        self._task: Optional[asyncio.Task] = None

    async def connect(self, timeout: float = 30.0) -> None:
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run(), name=f"mcp-{self.name}")
        try:
            await asyncio.wait_for(asyncio.shield(self._ready.wait()), timeout=timeout)
        except asyncio.TimeoutError:
            self._task.cancel()
            raise RuntimeError(f"MCP server '{self.name}' did not start within {timeout}s")
        if self._start_error:
            raise self._start_error

    async def _run(self) -> None:
        try:
            from mcp import ClientSession

            if self._config.get("type") == "http":
                await self._run_http(ClientSession)
            else:
                await self._run_stdio(ClientSession)
        except Exception as exc:
            log.error("MCP server '%s' connection error: %s", self.name, exc)
            self._start_error = exc
            self._ready.set()

    async def _run_stdio(self, ClientSession: Any) -> None:
        try:
            from mcp.client.stdio import stdio_client, StdioServerParameters
        except ImportError:
            from mcp.client.stdio import stdio_client  # type: ignore[no-redef]
            from mcp import StdioServerParameters  # type: ignore[no-redef]

        env = self._config.get("env") or None
        params = StdioServerParameters(
            command=self._config["command"],
            args=self._config.get("args", []),
            env=env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                log.info("MCP server '%s' ready (stdio)", self.name)
                await self._shutdown.wait()

    async def _run_http(self, ClientSession: Any) -> None:
        from mcp.client.streamable_http import streamable_http_client

        url = self._config["url"]
        headers = self._config.get("headers") or {}
        async with streamable_http_client(url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                log.info("MCP server '%s' ready (http)", self.name)
                await self._shutdown.wait()

    async def list_tools(self) -> List[Dict[str, Any]]:
        if not self._session:
            raise RuntimeError(f"MCP server '{self.name}' not connected")
        result = await self._session.list_tools()
        tools: List[Dict[str, Any]] = []
        for t in result.tools:
            schema: Dict[str, Any] = {}
            raw = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None) or {}
            if isinstance(raw, dict):
                schema = {k: v for k, v in raw.items() if k != "$schema"}
            tools.append({
                "type": "function",
                "function": {
                    "name": _make_tool_name(self.name, t.name),
                    "description": (getattr(t, "description", "") or "")[:512],
                    "parameters": schema or {"type": "object", "properties": {}},
                },
            })
        return tools

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        if not self._session:
            raise RuntimeError(f"MCP server '{self.name}' not connected")
        result = await self._session.call_tool(tool_name, args)
        parts: List[str] = []
        for block in result.content or []:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "data"):
                parts.append(f"[binary data, {len(block.data)} bytes]")
            else:
                parts.append(str(block))
        return "\n".join(parts) if parts else "(empty result)"

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()

    @property
    def is_alive(self) -> bool:
        return (
            self._task is not None
            and not self._task.done()
            and self._session is not None
        )


_LIST_TOOLS_TIMEOUT = 10.0   # seconds to wait for list_tools()
_CALL_TOOL_TIMEOUT  = 30.0   # seconds to wait for a single tool execution


class MCPClientManager:
    """
    Singleton that manages connections to all configured MCP servers.

    Connections are cached by (name, config) key so they survive across chat
    turns. A dead connection is evicted and transparently reconnected.
    Tool lists are cached per-session so repeated requests are instant.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, _ServerSession] = {}
        self._tool_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    def _key(self, name: str, config: Dict[str, Any]) -> str:
        return f"{name}::{json.dumps(config, sort_keys=True)}"

    async def _get_session(self, name: str, config: Dict[str, Any]) -> _ServerSession:
        key = self._key(name, config)
        async with self._lock:
            sess = self._sessions.get(key)
            if sess and sess.is_alive:
                return sess
            if sess:
                sess.stop()
                self._tool_cache.pop(key, None)
                del self._sessions[key]
            sess = _ServerSession(name, config)
            await sess.connect()
            self._sessions[key] = sess
            return sess

    async def _fetch_server_tools(
        self, name: str, config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Connect to one server and return its tool list (cached after first call)."""
        if not isinstance(config, dict) or ("command" not in config and "url" not in config):
            log.warning("MCP server '%s' missing 'command'/'url' — skipping", name)
            return []
        key = self._key(name, config)
        # Return cached tools if the session is still alive
        if key in self._tool_cache:
            sess = self._sessions.get(key)
            if sess and sess.is_alive:
                return self._tool_cache[key]
        try:
            sess = await self._get_session(name, config)
            tools = await asyncio.wait_for(sess.list_tools(), timeout=_LIST_TOOLS_TIMEOUT)
            self._tool_cache[key] = tools
            log.info("MCP '%s': %d tool(s) available", name, len(tools))
            return tools
        except asyncio.TimeoutError:
            log.error("MCP server '%s' list_tools timed out after %.0fs", name, _LIST_TOOLS_TIMEOUT)
            return []
        except Exception as exc:
            log.error("Could not load tools from MCP server '%s': %s", name, exc)
            return []

    async def get_tools(self, servers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Connect to all servers IN PARALLEL and return their tools as a merged
        OpenAI function-calling tool list. Servers that fail are skipped.
        """
        if not servers:
            return []

        results = await asyncio.gather(
            *[self._fetch_server_tools(name, config) for name, config in servers.items()]
        )
        return [tool for tools in results for tool in tools]

    async def call_tool(
        self, qualified_name: str, args: Dict[str, Any], servers: Dict[str, Any]
    ) -> str:
        """Execute a qualified MCP tool call (mcp__<server>__<tool>)."""
        parsed = _parse_tool_name(qualified_name)
        if not parsed:
            return f"Error: unrecognised MCP tool name '{qualified_name}'"
        server_name, tool_name = parsed
        config = servers.get(server_name)
        if not config:
            return f"Error: MCP server '{server_name}' not found in config"
        try:
            sess = await self._get_session(server_name, config)
            return await asyncio.wait_for(
                sess.call_tool(tool_name, args), timeout=_CALL_TOOL_TIMEOUT
            )
        except asyncio.TimeoutError:
            log.error("MCP tool '%s' timed out after %.0fs", qualified_name, _CALL_TOOL_TIMEOUT)
            return f"Error: MCP tool '{qualified_name}' timed out after {_CALL_TOOL_TIMEOUT:.0f}s"
        except Exception as exc:
            log.error("MCP tool call '%s' failed: %s", qualified_name, exc)
            return f"Error from MCP tool '{qualified_name}': {exc}"

    async def get_status(self, servers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Probe each server and return connection status with tool count."""
        results: List[Dict[str, Any]] = []
        for name, config in servers.items():
            if not isinstance(config, dict) or "command" not in config:
                results.append({"name": name, "connected": False, "error": "Invalid config"})
                continue
            try:
                sess = await self._get_session(name, config)
                tools = await asyncio.wait_for(sess.list_tools(), timeout=_LIST_TOOLS_TIMEOUT)
                results.append({"name": name, "connected": True, "toolCount": len(tools)})
            except asyncio.TimeoutError:
                results.append({"name": name, "connected": False, "error": "list_tools timed out"})
            except Exception as exc:
                results.append({"name": name, "connected": False, "error": str(exc)})
        return results

    @staticmethod
    def is_mcp_tool(name: str) -> bool:
        return name.startswith(_MCP_PREFIX)

    def stop_all(self) -> None:
        for sess in self._sessions.values():
            sess.stop()
        self._sessions.clear()


_manager: Optional[MCPClientManager] = None


def get_mcp_manager() -> MCPClientManager:
    global _manager
    if _manager is None:
        _manager = MCPClientManager()
    return _manager
