"""
llm agents for determining patient trial eligibility
"""

import json

from trial_project.api import generate_client
from trial_project.data.patients.evidence.generate_evidence import load_patient_evidence
from trial_project.data.trials.eligibility_verification import get_trial_eligibility_verification
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from trial_project.matching.check_eligibility_criteria import get_criterion_matching_prompt

client = generate_client()


class MatchedPatientEvidence(BaseModel):
  source_index: Literal[
    "demographics",
    "condition_index",
    "medication_index",
    "procedure_index",
    "observation_index",
    "encounter_index",
  ]
  normalized_name: str
  original_text: str
  date: str
  value: str
  units: str


class CriterionMatch(BaseModel):
  criterion_id: str
  criterion_type: Literal["inclusion", "exclusion"]
  criterion_text: str
  status: Literal[
    "meets",
    "does_not_meet",
    "insufficient_evidence",
    "excluded",
    "not_excluded",
  ]
  matched_patient_evidence: list[MatchedPatientEvidence] = Field(default_factory=list)
  possible_proxies: list[str] = Field(default_factory=list)
  missing_but_needed: list[str] = Field(default_factory=list)
  reasoning: str
  confidence: float = Field(ge=0.0, le=1.0)


class TrialEligibilityLLMResult(BaseModel):
  model_config = ConfigDict(extra="forbid")

  trial_id: str
  patient_id: str
  criterion_matches: list[CriterionMatch] = Field(default_factory=list)


class OverallTrialEligibilityLLMResult(BaseModel):
  model_config = ConfigDict(extra="forbid")

  trial_id: str
  patient_id: str
  overall_decision: Literal["eligible", "ineligible", "indeterminate"]
  overall_confidence_score: float = Field(ge=0.0, le=1.0)
  overall_rationale: str
  hard_stops: list[str] = Field(default_factory=list)
  manual_review_flags: list[str] = Field(default_factory=list)
  matching_notes: list[str] = Field(default_factory=list)


def _extract_trial_criterion_ids(trial_eligibility_json: str) -> list[str]:
  trial_payload = json.loads(trial_eligibility_json)
  atomic_criteria = trial_payload.get("atomic_criteria", [])
  return [
    str(item.get("criterion_id", "")).strip()
    for item in atomic_criteria
    if str(item.get("criterion_id", "")).strip()
  ]


def evaluate_trial_criteria_llm(
  patient_id,
  trial_id,
  model_name: str = "gpt-5-mini",
) -> TrialEligibilityLLMResult:

  trial_eligibility = get_trial_eligibility_verification(
    trial_id=trial_id,
    model_name=model_name,
    use_cache=True,
  )
  patient_evidence = load_patient_evidence(patient_id)
  if patient_evidence is None:
    raise ValueError(f"No patient evidence found for patient_id={patient_id}")

  input = f"Trial eligibility criteria: {trial_eligibility}\nPatient evidence profile: {patient_evidence}"

  response = client.responses.parse(
    model=model_name,
    instructions=get_criterion_matching_prompt(),
    input=input,
    text_format=TrialEligibilityLLMResult,
  )
  result = response.output_parsed
  if result is None:
    raise ValueError(
      f"Criterion matching returned no parsed output for patient_id={patient_id}, trial_id={trial_id}"
    )

  expected_criterion_ids = set(_extract_trial_criterion_ids(trial_eligibility))
  returned_criterion_ids = {
    criterion.criterion_id.strip()
    for criterion in result.criterion_matches
    if criterion.criterion_id.strip()
  }
  if expected_criterion_ids:
    missing_ids = sorted(expected_criterion_ids - returned_criterion_ids)
    if missing_ids:
      raise ValueError(
        "Criterion matching response is incomplete for "
        f"patient_id={patient_id}, trial_id={trial_id}. Missing criterion_ids={missing_ids}"
      )

  return result


def evaluate_overall_trial_eligibility_llm(
  patient_id: str,
  trial_id: str,
  criterion_matches: list[CriterionMatch],
  overall_prompt: str,
  model_name: str = "gpt-5-mini",
) -> OverallTrialEligibilityLLMResult:
  criterion_matches_json = json.dumps(
    [criterion.model_dump() for criterion in criterion_matches],
    ensure_ascii=True,
  )
  instructions = overall_prompt.replace(
    "{{criterion_matches_json}}",
    criterion_matches_json,
  )

  response = client.responses.parse(
    model=model_name,
    instructions=instructions,
    input=(
      f"Patient ID: {patient_id}\n"
      f"Trial ID: {trial_id}\n"
      f"Criterion count: {len(criterion_matches)}"
    ),
    text_format=OverallTrialEligibilityLLMResult,
  )
  result = response.output_parsed
  if result is None:
    raise ValueError(
      "Overall eligibility matching returned no parsed output for "
      f"patient_id={patient_id}, trial_id={trial_id}"
    )

  if result.patient_id != patient_id or result.trial_id != trial_id:
    raise ValueError(
      "Overall eligibility response IDs do not match request for "
      f"patient_id={patient_id}, trial_id={trial_id}"
    )

  return result
  