"""Intent-based prompt injection classifier using a local LLM."""

from __future__ import annotations

import json
from typing import Any

from langchain_ollama import ChatOllama


class IntentAgent:
    """Classify whether text appears to be a prompt injection attack."""

    def __init__(self) -> None:
        self.model = ChatOllama(model="llama3")

    def analyze(self, text: str) -> dict:
        """Analyze input text and classify prompt injection intent."""
        system_prompt = "You are a security classifier for prompt injection attacks."
        user_prompt = f"""Analyze the following input and determine if it is a prompt injection attack.

Input:
{text}

Respond in JSON format:
{{
"is_attack": true/false,
"confidence": number between 0 and 1,
"reason": short explanation
}}
"""

        response = self.model.invoke(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )

        content = response.content if isinstance(response.content, str) else str(response.content)

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(content[start : end + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

        return {
            "is_attack": False,
            "confidence": 0.0,
            "reason": "Parsing failed",
        }
