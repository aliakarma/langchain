"""Prompt injection detection middleware for agents."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class PromptInjectionGuardMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """Middleware for detecting prompt injection attacks before model execution."""

    def __init__(self, *, strategy: str = "block") -> None:
        super().__init__()
        self.strategy = strategy

    def _detect_prompt_injection(self, text: str) -> list[str]:
        patterns = [
            r"ignore (all|previous|prior) instructions",
            r"reveal (the )?(system prompt|hidden prompt)",
            r"disregard (the )?system prompt",
            r"you are now in developer mode",
        ]
        return [pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)]

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        for msg in messages:
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                continue

            matches = self._detect_prompt_injection(content)
            if matches:
                if self.strategy == "block":
                    raise ValueError(
                        "Prompt injection detected in input messages. "
                        f"Matched patterns: {matches}"
                    )
                if self.strategy == "annotate":
                    return {"prompt_injection_detected": True}

        return None
