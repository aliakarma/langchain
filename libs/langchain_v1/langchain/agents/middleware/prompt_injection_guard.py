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
            r"ignore.*instructions",
            r"(reveal|show|display).*(system prompt|hidden prompt)",
            r"disregard.*instructions",
            r"developer mode",
            r"bypass.*safety",
            r"override.*rules",
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

        for content in contents:
            segments = re.split(r"[.!?]\s+", content)
            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue

                keyword_groups = [
                    ["ignore", "instructions"],
                    ["reveal", "system", "prompt"],
                    ["show", "system", "prompt"],
                    ["hidden", "prompt"],
                    ["developer", "mode"],
                ]
                lower_segment = segment.lower()
                matched_groups = [
                    group
                    for group in keyword_groups
                    if all(word in lower_segment for word in group)
                ]
                if matched_groups:
                    matched_keywords = [" ".join(group) for group in matched_groups]
                    if self.strategy == "block":
                        raise ValueError(
                            "Prompt injection detected in input messages. "
                            f"Detected patterns: {matched_keywords}"
                        )
                    if self.strategy == "annotate":
                        return {
                            "prompt_injection_detected": True,
                            "detected_patterns": matched_keywords,
                        }

                segment_matches = self._detect_prompt_injection(segment)
                if segment_matches:
                    all_matches.update(segment_matches)
                    if self.strategy == "block":
                        raise ValueError(
                            "Prompt injection detected in input messages. "
                            f"Detected patterns: {sorted(all_matches)}"
                        )
                    if self.strategy == "annotate":
                        return {
                            "prompt_injection_detected": True,
                            "detected_patterns": sorted(all_matches),
                        }

                if self.use_intent_agent:
                    result = self.intent_agent.analyze(segment)
                    is_attack = result.get("is_attack") is True
                    confidence = result.get("confidence", 0.0)

                    try:
                        confidence_value = float(confidence)
                    except (TypeError, ValueError):
                        confidence_value = 0.0

                    if is_attack and confidence_value >= 0.7:
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
