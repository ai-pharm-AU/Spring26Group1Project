import json
from typing import Literal

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field

from trial_project.api import generate_client
from trial_project.context import data_dir
from trial_project.data.patients.evidence.generate_evidence import get_patient_evidence
from trial_project.data.patients.evidence.schema import PatientEvidence
from trial_project.data.patients.load_patient import get_patient_llm_json

verification_path = data_dir / "processed_data" / "patient_evidence_verification.parquet"

_VERIFICATION_COLUMNS = [
  "patient_id",
  "patient_evidence_verification",
  "corrected_patient_evidence",
]


class RawEvidence(BaseModel):
  model_config = ConfigDict(extra="forbid")

  source_section: Literal[
    "patient",
    "encounters",
    "conditions",
    "medications",
    "observations",
    "procedures",
    "allergies",
    "immunizations",
    "careplans",
  ] = "patient"
  original_text: str = ""
  date_or_start: str = ""
  end_date: str = ""
  value: str = ""
  units: str = ""


class ClaimReview(BaseModel):
  model_config = ConfigDict(extra="forbid")

  claim_id: str = ""
  original_claim: str = ""
  claim_type: Literal[
    "demographic",
    "condition",
    "medication",
    "procedure",
    "observation",
    "encounter",
    "allergy",
    "immunization",
    "careplan",
    "other",
  ] = "other"
  verification_status: Literal[
    "supported",
    "partially_supported",
    "contradicted",
    "not_found",
  ] = "not_found"
  verification_explanation: str = ""
  matched_raw_evidence: list[RawEvidence] = Field(default_factory=list)
  corrected_claim: str = ""
  is_proxy_only: bool = False
  should_be_used_for_eligibility: bool = True
  confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class OmittedRelevantEvidence(BaseModel):
  model_config = ConfigDict(extra="forbid")

  related_claim_id: str = ""
  why_relevant: str = ""
  raw_evidence: list[RawEvidence] = Field(default_factory=list)
  suggested_claim: str = ""


class ContradictoryPatientEvidence(BaseModel):
  model_config = ConfigDict(extra="forbid")

  claim_id: str = ""
  description: str = ""
  raw_evidence: list[RawEvidence] = Field(default_factory=list)


class PatientEvidenceVerification(BaseModel):
  model_config = ConfigDict(extra="forbid")

  patient_id: str = ""
  overall_verification_status: Literal[
    "fully_verified",
    "partially_verified",
    "verification_failed",
    "manual_review_needed",
  ] = "manual_review_needed"
  overall_verification_summary: str = ""
  fit_for_matching: bool = False
  fit_for_matching_explanation: str = ""
  corrected_patient_evidence: PatientEvidence
  claim_reviews: list[ClaimReview] = Field(default_factory=list)
  omitted_relevant_evidence: list[OmittedRelevantEvidence] = Field(default_factory=list)
  unsupported_or_problematic_claims: list[str] = Field(default_factory=list)
  contradictory_patient_evidence: list[ContradictoryPatientEvidence] = Field(default_factory=list)
  manual_review_flags: list[str] = Field(default_factory=list)
  verification_notes: list[str] = Field(default_factory=list)


def _empty_verification_frame() -> DataFrame:
  return DataFrame(columns=_VERIFICATION_COLUMNS)


def load_all_patient_evidence_verification() -> DataFrame:
  """Load all cached patient evidence verification rows."""
  if not verification_path.exists():
    return _empty_verification_frame()

  verification_df = pd.read_parquet(verification_path)
  for col in _VERIFICATION_COLUMNS:
    if col not in verification_df.columns:
      verification_df[col] = pd.NA
  return verification_df


def load_patient_evidence_verification(
  patient_id: str,
  verification_df: DataFrame | None = None,
) -> str | None:
  """Load cached corrected patient evidence JSON text for one patient."""
  if verification_df is None:
    verification_df = load_all_patient_evidence_verification()

  patient_rows = verification_df[verification_df["patient_id"] == patient_id]
  if patient_rows.empty:
    return None

  corrected_values = patient_rows["corrected_patient_evidence"].dropna()
  if corrected_values.empty:
    return None

  return str(corrected_values.iloc[0])


def save_patient_evidence_verification(
  patient_id: str,
  patient_evidence_verification: str,
  corrected_patient_evidence: str,
) -> None:
  """Upsert one patient evidence verification row into the parquet cache."""
  verification_df = load_all_patient_evidence_verification()

  for col in _VERIFICATION_COLUMNS:
    if col not in verification_df.columns:
      verification_df[col] = pd.NA

  mask = verification_df["patient_id"] == patient_id
  new_row = {
    "patient_id": patient_id,
    "patient_evidence_verification": patient_evidence_verification,
    "corrected_patient_evidence": corrected_patient_evidence,
  }

  verification_df = verification_df.loc[~mask]
  verification_df = pd.concat([verification_df, DataFrame([new_row])], ignore_index=True)

  verification_path.parent.mkdir(parents=True, exist_ok=True)
  verification_df.to_parquet(verification_path, index=False)


def _patient_evidence_verification_prompt() -> str:
  return """You are verifying and correcting extracted patient evidence against the raw patient record.

Goal:

Verify whether the extracted patient evidence is accurate to the patient record and produce corrected patient evidence that is ready for downstream matching.

Important rules:

- The raw patient record is the only source of truth for patient facts.
- Do not invent evidence.
- Do not treat missing evidence as negative evidence.
- Preserve dates, units, active vs historical status, negation, and temporality exactly.
- If extracted evidence is partly correct but overstated, mark it as partially_supported and explain the overstatement.
- If extracted evidence is absent from the patient record, mark it as not_found.
- If the patient record contains contradictory evidence, mark it as contradicted.

Patient record sections may include:
- patient
- encounters
- conditions
- medications
- observations
- procedures
- allergies
- immunizations
- careplans

Instructions:

1. Read the extracted patient evidence under review.
2. Read the raw patient record and treat it as the source of truth.
3. Verify each extracted evidence item against the raw patient record.
4. For each item, return:
   - verification_status
   - a short explanation
   - the best matching raw record evidence, if any
   - a corrected version of the claim when the original is inaccurate or overstated
5. If extracted evidence omits an important patient fact clearly present in the raw record, add it under omitted_relevant_evidence.
6. Build corrected_patient_evidence as a complete corrected object that conforms to the PatientEvidence schema.
7. Keep the output compact and evidence-based.
8. Return JSON only.

Use the provided JSON schema for output shape and constraints.

Input:

{{verification_input_json}}
"""


def _verification_input(
  patient_id: str,
  patient_evidence: str,
  patient_record_json: str,
) -> str:
  payload = {
    "patient_id": patient_id,
    "patient_evidence_under_review": json.loads(patient_evidence),
    "raw_patient_record": json.loads(patient_record_json),
  }
  return json.dumps(payload, ensure_ascii=True)


def generate_patient_evidence_verification(
  patient_id: str,
  model_name: str = "gpt-5-mini",
) -> tuple[str, str]:
  """Generate verification JSON and corrected patient evidence JSON for one patient."""
  client = generate_client()
  patient_evidence = get_patient_evidence(patient_id)
  patient_record_json = get_patient_llm_json(patient_id)
  verification_input = _verification_input(
    patient_id=patient_id,
    patient_evidence=patient_evidence,
    patient_record_json=patient_record_json,
  )

  response = client.responses.parse(
    model=model_name,
    instructions=_patient_evidence_verification_prompt(),
    input=verification_input,
    text_format=PatientEvidenceVerification,
  )
  validated = PatientEvidenceVerification.model_validate_json(response.output_text)
  verification_json = json.dumps(validated.model_dump(), ensure_ascii=True)
  corrected_json = json.dumps(validated.corrected_patient_evidence.model_dump(), ensure_ascii=True)
  return verification_json, corrected_json


def get_patient_evidence_verification(
  patient_id: str,
  model_name: str = "gpt-5-mini",
  use_cache: bool = True,
) -> str:
  """Return corrected patient evidence JSON (cached when available)."""
  if use_cache:
    verification_df = load_all_patient_evidence_verification()
    existing = load_patient_evidence_verification(patient_id, verification_df)
    if existing is not None:
      return existing

  verification, corrected = generate_patient_evidence_verification(
    patient_id=patient_id,
    model_name=model_name,
  )
  save_patient_evidence_verification(
    patient_id=patient_id,
    patient_evidence_verification=verification,
    corrected_patient_evidence=corrected,
  )
  return corrected
