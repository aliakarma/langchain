"""Prompt injection detection middleware for agents."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from langchain.agents.security.intent_agent import IntentAgent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class PromptInjectionGuardMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """Middleware for detecting prompt injection attacks before model execution."""

    def __init__(
        self,
        *,
        strategy: str = "block",
        use_intent_agent: bool = True,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.use_intent_agent = use_intent_agent
        self.intent_agent = IntentAgent()

    def _detect_prompt_injection(self, text: str) -> list[str]:
        patterns = [
            r"ignore (all|previous|prior) instructions",
            r"reveal (the )?(system prompt|hidden prompt)",
            r"disregard (the )?system prompt",
            r"you are now in developer mode",
        ]
        return [pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)]

    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        all_matches: set[str] = set()
        contents: list[str] = []

        for msg in messages:
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                continue

            contents.append(content)

            matches = self._detect_prompt_injection(content)
            all_matches.update(matches)
            if matches and self.strategy == "block":
                raise ValueError(
                    "Prompt injection detected in input messages. "
                    f"Detected patterns: {sorted(all_matches)}"
                )

        detected_patterns = sorted(all_matches)
        if detected_patterns:
            if self.strategy == "annotate":
                return {
                    "prompt_injection_detected": True,
                    "detected_patterns": detected_patterns,
                }

        semantic_detected = False
        if not detected_patterns and self.use_intent_agent:
            for content in contents:
                result = self.intent_agent.analyze(content)
                is_attack = result.get("is_attack") is True
                confidence = result.get("confidence", 0.0)

                try:
                    confidence_value = float(confidence)
                except (TypeError, ValueError):
                    confidence_value = 0.0

                if is_attack and confidence_value >= 0.7:
                    semantic_detected = True
                    break

        if semantic_detected:
            if self.strategy == "block":
                raise ValueError(
                    "Prompt injection detected in input messages. "
                    "semantic detection triggered"
                )
            if self.strategy == "annotate":
                return {
                    "prompt_injection_detected": True,
                    "detected_patterns": ["semantic detection triggered"],
                }

        return None
