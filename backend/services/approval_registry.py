import asyncio
from typing import Any, Dict, Optional


class ApprovalRegistry:
    """
    Lets agent coroutines pause and wait for out-of-band user approvals.

    Flow:
      1. Agent calls register(pending_id)  — before yielding the SSE event
      2. Agent calls await wait_for(pending_id) — suspends the coroutine
      3. Extension calls /edit/apply or /terminal/execute
      4. Those endpoints call resolve(pending_id, result)
      5. Agent resumes with the result dict
    """

    def __init__(self) -> None:
        self._events: Dict[str, asyncio.Event] = {}
        self._results: Dict[str, Any] = {}

    def register(self, pending_id: str) -> None:
        """Register a pending ID before yielding its SSE event."""
        self._events[pending_id] = asyncio.Event()

    def resolve(self, pending_id: str, result: Any) -> bool:
        """
        Called by apply/execute endpoints to unblock a waiting agent.
        Returns True if an agent was waiting, False if not (standalone use).
        """
        event = self._events.get(pending_id)
        if event is None:
            return False
        self._results[pending_id] = result
        event.set()
        return True

    async def wait_for(self, pending_id: str, timeout: float = 300.0) -> Optional[Any]:
        """
        Suspend until resolved or timeout.
        Cleans up internal state in all cases.
        """
        event = self._events.get(pending_id)
        if event is None:
            return None

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            self._events.pop(pending_id, None)

        return self._results.pop(pending_id, None)

    def is_registered(self, pending_id: str) -> bool:
        return pending_id in self._events


_instance: Optional[ApprovalRegistry] = None


def get_approval_registry() -> ApprovalRegistry:
    global _instance
    if _instance is None:
        _instance = ApprovalRegistry()
    return _instance
