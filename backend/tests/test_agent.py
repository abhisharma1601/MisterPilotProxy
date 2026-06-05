"""
Tests for the ReAct agent loop.

Strategy: mock get_deepseek_client() so no real LLM calls are made.
The mocked client returns either a plain text response or a tool-call response
depending on the test scenario.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents import react_agent
from backend.services.approval_registry import ApprovalRegistry


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_choice(
    content: str = "",
    finish_reason: str = "stop",
    tool_calls: List[Dict] | None = None,
):
    """Build a fake ChatCompletion choice object."""
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    if tool_calls:
        tcs = []
        for tc in tool_calls:
            m = MagicMock()
            m.id = tc["id"]
            m.function.name = tc["name"]
            m.function.arguments = json.dumps(tc["args"])
            tcs.append(m)
        choice.message.tool_calls = tcs
    else:
        choice.message.tool_calls = None
    return choice


def _make_response(
    content: str = "",
    finish_reason: str = "stop",
    tool_calls: List[Dict] | None = None,
):
    resp = MagicMock()
    resp.choices = [_make_choice(content, finish_reason, tool_calls)]
    return resp


async def _collect(gen: AsyncIterator[Dict]) -> List[Dict]:
    return [event async for event in gen]


# ── ApprovalRegistry unit tests ───────────────────────────────────────────────

class TestApprovalRegistry:
    def test_resolve_returns_false_when_not_registered(self):
        reg = ApprovalRegistry()
        assert reg.resolve("nonexistent", {"applied": True}) is False

    def test_resolve_returns_true_when_registered(self):
        reg = ApprovalRegistry()
        reg.register("abc")
        assert reg.resolve("abc", {"applied": True}) is True

    @pytest.mark.asyncio
    async def test_wait_for_receives_result(self):
        reg = ApprovalRegistry()
        reg.register("x")

        async def _resolve_soon():
            await asyncio.sleep(0.01)
            reg.resolve("x", {"applied": True})

        asyncio.create_task(_resolve_soon())
        result = await reg.wait_for("x", timeout=1.0)
        assert result == {"applied": True}

    @pytest.mark.asyncio
    async def test_wait_for_timeout_returns_none(self):
        reg = ApprovalRegistry()
        reg.register("y")
        result = await reg.wait_for("y", timeout=0.05)
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_unknown_id_returns_none(self):
        reg = ApprovalRegistry()
        result = await reg.wait_for("unknown", timeout=0.1)
        assert result is None

    def test_is_registered(self):
        reg = ApprovalRegistry()
        assert not reg.is_registered("a")
        reg.register("a")
        assert reg.is_registered("a")


# ── react_agent.run — simple text response ────────────────────────────────────

class TestAgentSimpleText:
    @pytest.mark.asyncio
    async def test_plain_answer_yields_chunk_then_done(self):
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(
            return_value=_make_response("Hello from the agent!")
        )

        with patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "hi"}], workspace_root=None)
            )

        types = [e["type"] for e in events]
        assert types == ["chunk", "done"]
        chunks = [e["content"] for e in events if e["type"] == "chunk"]
        assert "Hello from the agent!" in chunks[0]

    @pytest.mark.asyncio
    async def test_no_workspace_root_mentions_it(self):
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(
            return_value=_make_response("ok")
        )

        with patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "hi"}], workspace_root=None)
            )

        # System prompt should mention no workspace — verify by checking call args
        call_args = mock_llm.complete_with_tools.call_args
        messages = call_args[0][0]
        system_text = messages[0]["content"]
        assert "No workspace" in system_text


# ── read_file tool ────────────────────────────────────────────────────────────

class TestReadFileTool:
    @pytest.mark.asyncio
    async def test_read_file_injects_content(self, tmp_path: Path):
        (tmp_path / "hello.txt").write_text("line1\nline2\n")

        responses = [
            _make_response(
                content="Let me read the file.",
                finish_reason="tool_calls",
                tool_calls=[{"id": "tc1", "name": "read_file", "args": {"path": "hello.txt"}}],
            ),
            _make_response("The file has two lines.", finish_reason="stop"),
        ]
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=responses)

        with patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "read it"}], str(tmp_path))
            )

        types = [e["type"] for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "done" in types

        tool_result_event = next(e for e in events if e["type"] == "tool_result")
        assert "hello.txt" in tool_result_event["tool"] or "1:" in tool_result_event["content"]

        # Second LLM call should include the tool result in context
        second_call_messages = mock_llm.complete_with_tools.call_args_list[1][0][0]
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "line1" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_read_file_path_traversal_returns_error(self, tmp_path: Path):
        responses = [
            _make_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[{"id": "tc1", "name": "read_file", "args": {"path": "../../etc/passwd"}}],
            ),
            _make_response("sorry", finish_reason="stop"),
        ]
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=responses)

        with patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "read it"}], str(tmp_path))
            )

        tool_result_event = next(e for e in events if e["type"] == "tool_result")
        assert "error" in tool_result_event["content"].lower() or "traversal" in tool_result_event["content"].lower()


# ── list_files tool ───────────────────────────────────────────────────────────

class TestListFilesTool:
    @pytest.mark.asyncio
    async def test_list_files_returns_paths(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")

        responses = [
            _make_response(
                finish_reason="tool_calls",
                tool_calls=[{"id": "tc1", "name": "list_files", "args": {}}],
            ),
            _make_response("Found two files."),
        ]
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=responses)

        with patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm):
            await _collect(
                react_agent.run([{"role": "user", "content": "list"}], str(tmp_path))
            )

        tool_msgs = [
            m for m in mock_llm.complete_with_tools.call_args_list[1][0][0]
            if m.get("role") == "tool"
        ]
        content = tool_msgs[0]["content"]
        assert "a.py" in content and "b.py" in content


# ── search_code tool ──────────────────────────────────────────────────────────

class TestSearchCodeTool:
    @pytest.mark.asyncio
    async def test_search_finds_text(self, tmp_path: Path):
        (tmp_path / "src.py").write_text("def hello_world(): pass\n")

        responses = [
            _make_response(
                finish_reason="tool_calls",
                tool_calls=[{"id": "tc1", "name": "search_code", "args": {"query": "hello_world"}}],
            ),
            _make_response("Found it."),
        ]
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=responses)

        with patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm):
            await _collect(
                react_agent.run([{"role": "user", "content": "find it"}], str(tmp_path))
            )

        tool_msgs = [
            m for m in mock_llm.complete_with_tools.call_args_list[1][0][0]
            if m.get("role") == "tool"
        ]
        assert "hello_world" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_search_no_results_message(self, tmp_path: Path):
        (tmp_path / "src.py").write_text("pass\n")

        responses = [
            _make_response(
                finish_reason="tool_calls",
                tool_calls=[{"id": "tc1", "name": "search_code", "args": {"query": "xyznotfound"}}],
            ),
            _make_response("Not found."),
        ]
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=responses)

        with patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm):
            await _collect(
                react_agent.run([{"role": "user", "content": "find"}], str(tmp_path))
            )

        tool_msgs = [
            m for m in mock_llm.complete_with_tools.call_args_list[1][0][0]
            if m.get("role") == "tool"
        ]
        assert "No matches" in tool_msgs[0]["content"]


# ── write_file tool (approval) ────────────────────────────────────────────────

class TestWriteFileTool:
    @pytest.mark.asyncio
    async def test_write_file_emits_pending_edit_and_waits(self, tmp_path: Path):
        responses = [
            _make_response(
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"path": "out.txt", "content": "hello"},
                }],
            ),
            _make_response("Written."),
        ]
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=responses)

        # Simulate the extension calling /edit/apply after 50ms
        registry = ApprovalRegistry()

        async def _approve():
            await asyncio.sleep(0.05)
            # Register the pending edit ID — we need to know it first.
            # In the real flow the agent registers and then yields; we monitor
            # the registry to find the ID.
            while not registry._events:
                await asyncio.sleep(0.005)
            pending_id = next(iter(registry._events))
            registry.resolve(pending_id, {"applied": True, "path": "out.txt"})

        task = asyncio.create_task(_approve())

        with (
            patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm),
            patch("backend.agents.react_agent.get_approval_registry", return_value=registry),
        ):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "write it"}], str(tmp_path))
            )

        await task

        pending_events = [e for e in events if e["type"] == "pending_edit"]
        assert len(pending_events) == 1
        assert pending_events[0]["path"] == "out.txt"

        # Tool result should say file was written
        tool_result_event = next(e for e in events if e["type"] == "tool_result")
        assert "written" in tool_result_event["content"].lower() or "out.txt" in tool_result_event["content"]

    @pytest.mark.asyncio
    async def test_write_file_rejected_reports_back(self, tmp_path: Path):
        responses = [
            _make_response(
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"path": "out.txt", "content": "hello"},
                }],
            ),
            _make_response("Understood, not written."),
        ]
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=responses)

        registry = ApprovalRegistry()

        async def _reject():
            while not registry._events:
                await asyncio.sleep(0.005)
            pending_id = next(iter(registry._events))
            registry.resolve(pending_id, {"applied": False, "path": "out.txt"})

        task = asyncio.create_task(_reject())

        with (
            patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm),
            patch("backend.agents.react_agent.get_approval_registry", return_value=registry),
        ):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "write it"}], str(tmp_path))
            )

        await task

        tool_result_event = next(e for e in events if e["type"] == "tool_result")
        assert "rejected" in tool_result_event["content"].lower()


# ── execute_terminal tool ─────────────────────────────────────────────────────

class TestExecuteTerminalTool:
    @pytest.mark.asyncio
    async def test_terminal_approved_resolves_with_output(self, tmp_path: Path):
        responses = [
            _make_response(
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "tc1",
                    "name": "execute_terminal",
                    "args": {"command": "echo test"},
                }],
            ),
            _make_response("Command ran."),
        ]
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=responses)

        registry = ApprovalRegistry()

        async def _approve():
            while not registry._events:
                await asyncio.sleep(0.005)
            pending_id = next(iter(registry._events))
            registry.resolve(pending_id, {
                "approved": True,
                "stdout": "test\n",
                "stderr": "",
                "exit_code": 0,
                "duration_ms": 12,
                "timed_out": False,
            })

        task = asyncio.create_task(_approve())

        with (
            patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm),
            patch("backend.agents.react_agent.get_approval_registry", return_value=registry),
        ):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "run it"}], str(tmp_path))
            )

        await task

        pending = [e for e in events if e["type"] == "pending_terminal"]
        assert len(pending) == 1
        assert pending[0]["command"] == "echo test"

        tool_result = next(e for e in events if e["type"] == "tool_result")
        assert "test" in tool_result["content"] or "exit code: 0" in tool_result["content"]

    @pytest.mark.asyncio
    async def test_terminal_denied_reports_back(self, tmp_path: Path):
        responses = [
            _make_response(
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "tc1",
                    "name": "execute_terminal",
                    "args": {"command": "rm -rf /"},
                }],
            ),
            _make_response("Noted."),
        ]
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=responses)

        registry = ApprovalRegistry()

        async def _deny():
            while not registry._events:
                await asyncio.sleep(0.005)
            pending_id = next(iter(registry._events))
            registry.resolve(pending_id, {"approved": False})

        task = asyncio.create_task(_deny())

        with (
            patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm),
            patch("backend.agents.react_agent.get_approval_registry", return_value=registry),
        ):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "delete everything"}], str(tmp_path))
            )

        await task

        tool_result = next(e for e in events if e["type"] == "tool_result")
        assert "denied" in tool_result["content"].lower()


# ── guard rails ───────────────────────────────────────────────────────────────

class TestAgentGuards:
    @pytest.mark.asyncio
    async def test_max_iterations_emits_error(self, tmp_path: Path):
        # Always returns tool_calls → infinite loop → should hit MAX_ITERATIONS
        infinite_response = _make_response(
            finish_reason="tool_calls",
            tool_calls=[{"id": "tc1", "name": "list_files", "args": {}}],
        )
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(return_value=infinite_response)

        with patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "loop"}], str(tmp_path))
            )

        error_events = [e for e in events if e["type"] == "error"]
        assert error_events, "Expected an error event after max iterations"
        assert "iterations" in error_events[0]["message"].lower()
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_llm_error_yields_error_then_done(self):
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=RuntimeError("LLM offline"))

        with patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "hi"}], workspace_root=None)
            )

        assert events[0]["type"] == "error"
        assert "LLM offline" in events[0]["message"]
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_content(self, tmp_path: Path):
        responses = [
            _make_response(
                finish_reason="tool_calls",
                tool_calls=[{"id": "tc1", "name": "nonexistent_tool", "args": {}}],
            ),
            _make_response("ok"),
        ]
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=responses)

        with patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "call it"}], str(tmp_path))
            )

        tool_result = next(e for e in events if e["type"] == "tool_result")
        assert "Unknown tool" in tool_result["content"]

    @pytest.mark.asyncio
    async def test_file_tools_unavailable_without_workspace(self):
        responses = [
            _make_response(
                finish_reason="tool_calls",
                tool_calls=[{"id": "tc1", "name": "read_file", "args": {"path": "x.py"}}],
            ),
            _make_response("no workspace"),
        ]
        mock_llm = MagicMock()
        mock_llm.complete_with_tools = AsyncMock(side_effect=responses)

        with patch("backend.agents.react_agent.get_deepseek_client", return_value=mock_llm):
            events = await _collect(
                react_agent.run([{"role": "user", "content": "read"}], workspace_root=None)
            )

        tool_result = next(e for e in events if e["type"] == "tool_result")
        assert "unavailable" in tool_result["content"].lower()
