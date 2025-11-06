"""Prompt templates and few-shot examples for the triage agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain.prompts import FewShotPromptTemplate, PromptTemplate


@dataclass(frozen=True)
class TriageExample:
    """Represents a single few-shot triage example."""

    symptoms: str
    reasoning: str
    severity: str
    recommendation: str

    def format(self) -> str:
        return (
            "<example>\n"
            f"<symptoms>{self.symptoms}</symptoms>\n"
            f"<reasoning>{self.reasoning}</reasoning>\n"
            f"<urgency>{self.severity}</urgency>\n"
            f"<plan>{self.recommendation}</plan>\n"
            "</example>"
        )


def default_examples() -> List[TriageExample]:
    """Return curated few-shot examples for the agent."""

    return [
        TriageExample(
            symptoms=(
                "Headache for two days, mild fever (100F), body aches. No breathing "
                "issues. Drinking fluids and can eat."
            ),
            reasoning=(
                "Symptoms consistent with viral infection without red flags. Fever is "
                "low-grade and patient remains hydrated."
            ),
            severity="Self-care",
            recommendation=(
                "Rest, hydrate, take paracetamol for fever, monitor for worsening "
                "symptoms such as high fever or shortness of breath."
            ),
        ),
        TriageExample(
            symptoms=(
                "Elderly patient with chest tightness, sweating, and pain radiating to "
                "left arm for 20 minutes."
            ),
            reasoning=(
                "Possible acute coronary syndrome; chest pain with radiation and "
                "autonomic symptoms is an emergency."
            ),
            severity="Emergency",
            recommendation=(
                "Call emergency services immediately, chew aspirin if not allergic, and "
                "do not delay transport to hospital."
            ),
        ),
        TriageExample(
            symptoms=(
                "Child with cough, sore throat, mild fever (101F), eating less but able "
                "to drink, no breathing difficulty."
            ),
            reasoning=(
                "Likely upper respiratory infection without danger signs; evaluation at "
                "primary care can ensure no complications."
            ),
            severity="BHU Visit",
            recommendation=(
                "Schedule visit to Basic Health Unit within 24 hours, continue fluids, "
                "and use honey/lemon for cough if age >1 year."
            ),
        ),
    ]


def build_prompt() -> FewShotPromptTemplate:
    """Construct the FewShotPromptTemplate used by the agent."""

    example_prompt = PromptTemplate(
        input_variables=["symptoms", "reasoning", "severity", "recommendation"],
        template=(
            "<example>\n"
            "<symptoms>{symptoms}</symptoms>\n"
            "<reasoning>{reasoning}</reasoning>\n"
            "<urgency>{severity}</urgency>\n"
            "<plan>{recommendation}</plan>\n"
            "</example>"
        ),
    )

    prompt = FewShotPromptTemplate(
        prefix=(
            "You are the Triage Agent for a citizen-facing telehealth service."
            "\nYou converse in Urdu and English, accept free-text symptom "
            "descriptions, and assign an urgency level: Self-care, BHU Visit, or "
            "Emergency.\n"
            "Use structured XML tags in your final answer with sections for "
            "reasoning, urgency, and plan."
            "\nFollow this process:\n"
            "1. Understand the key symptoms, onset, and risk factors.\n"
            "2. Determine the most likely urgency level and justify it.\n"
            "3. Provide actionable next steps, including safety net advice.\n"
            "You may respond in the user's language when possible."
            "\nOutput format instructions:\n{format_instructions}\n"
        ),
        examples=[
            {
                "symptoms": ex.symptoms,
                "reasoning": ex.reasoning,
                "severity": ex.severity,
                "recommendation": ex.recommendation,
            }
            for ex in default_examples()
        ],
        example_prompt=example_prompt,
        suffix=(
            "<conversation>\n"
            "<symptoms>{symptoms}</symptoms>\n"
            "</conversation>\n"
            "Provide your assessment now."
        ),
        input_variables=["symptoms"],
    )

    return prompt
