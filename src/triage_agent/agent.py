"""Triage agent implementation built with LangChain and Gemini."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from .prompt import build_prompt


class TriageAssessment(BaseModel):
    """Structured response returned by the agent."""

    reasoning: str = Field(..., description="Explanation for the urgency decision")
    urgency: str = Field(..., description="One of: Self-care, BHU Visit, Emergency")
    plan: str = Field(..., description="Follow-up steps and safety advice")


@dataclass
class TriageAgent:
    """High-level interface for assessing symptoms."""

    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.2
    safety_settings: Optional[dict] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple runtime validation
        if not os.getenv("GOOGLE_API_KEY"):
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is required for Gemini access."
            )

        parser = PydanticOutputParser(pydantic_object=TriageAssessment)
        prompt = build_prompt()

        self._prompt = prompt.partial(
            format_instructions=parser.get_format_instructions()
        )
        self._parser = parser
        self._llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            safety_settings=self.safety_settings,
        )

    def assess(self, symptoms: str) -> TriageAssessment:
        """Assess a symptom description and return a structured triage plan."""

        if not symptoms.strip():
            raise ValueError("Symptoms text must not be empty.")

        chain = self._prompt | self._llm | self._parser
        return chain.invoke({"symptoms": symptoms})
