"""LLM helpers for ranking patient-trial matches."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trial_project.api import generate_client

client = generate_client()


class TrialRankingLLMResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_relevance_score: float = Field(ge=0.0, le=100.0)
    potential_benefit_score: float = Field(ge=0.0, le=100.0)
    safety_score: float = Field(ge=0.0, le=100.0)
    evidence_strength_score: float = Field(ge=0.0, le=100.0)
    feasibility_score: float = Field(ge=0.0, le=100.0)
    overall_score: float = Field(ge=0.0, le=100.0)


ranking_prompt = """
You are ranking an eligible patient-clinical trial pair according to various factors after eligibility matching has already been performed.

Goal:

Given one structured trial profile, one structured patient profile, and the existing trial-patient matching output, assign a ranking score that prioritizes how suitable and worthwhile this trial is for this patient.

Important rules:

- This is a ranking score, not an eligibility decision.
- Do not redo inclusion/exclusion matching from scratch.
- Use the prior matching output as the source of truth for eligibility status, unresolved criteria, and match confidence.
- Rank based on likely clinical relevance, potential for patient benefit, safety/risk concerns, evidence strength, and practical fit.
- Be conservative when evidence is weak or incomplete.
- If overall_decision is ineligible, cap overall_score at 15 unless explicitly instructed otherwise.
- If overall_decision is indeterminate, subtract 10 to 25 points depending on how many key criteria are unresolved.
- Keep reasoning short and evidence-based.
- Return JSON only.

Scoring framework:

Assign the following subscores from 0 to 100:

1. condition_relevance_score
   - How directly the trial addresses the patient’s active condition(s) or priority clinical problem

2. potential_benefit_score
   - How plausible it is that the trial could positively impact the patient’s wellbeing or disease course

3. safety_score
   - How favorable the patient’s safety/risk profile is for this trial
   - Higher = safer / fewer concerns
   - Lower = more risk concerns

4. evidence_strength_score
   - How strong and direct the underlying matching evidence is

5. feasibility_score
   - How practical participation appears, if feasibility information is available

Compute the final overall_score from 0 to 100 using this weighted formula:

- condition_relevance_score * 0.35
- potential_benefit_score * 0.25
- safety_score * 0.25
- evidence_strength_score * 0.10
- feasibility_score * 0.05

Then apply the adjustment rules above for ineligible or indeterminate pairs.

Return JSON in this exact shape:

{
	"condition_relevance_score": 0,
	"potential_benefit_score": 0,
	"safety_score": 0,
	"evidence_strength_score": 0,
	"feasibility_score": 0,
	"overall_score": 0
}
"""


def _stringify_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()

    return json.dumps(payload, ensure_ascii=True, indent=2, default=str)


def evaluate_trial_ranking_llm(
    trial_profile: Any,
    patient_profile: Any,
    matching_results: Any,
    model_name: str = "gpt-5-mini",
) -> TrialRankingLLMResult:
    """Rank a single patient-trial pair using the criteria model."""
    trial_profile_json = _stringify_payload(trial_profile)
    patient_profile_json = _stringify_payload(patient_profile)
    matching_results_json = _stringify_payload(matching_results)

    instructions = (
        ranking_prompt.strip()
        + "\n\nTrial_profile:\n"
        + trial_profile_json
        + "\n\nPatient_profile:\n"
        + patient_profile_json
        + "\n\nMatching_results:\n"
        + matching_results_json
    )

    response = client.responses.parse(
        model=model_name,
        instructions=instructions,
        input=(
            "Rank the patient-trial pair using the provided trial profile, patient profile, "
            "and matching results."
        ),
        text_format=TrialRankingLLMResult,
    )

    result = response.output_parsed
    if result is None:
        raise ValueError(
            "Ranking returned no parsed output for the provided patient-trial pair"
        )

    return result