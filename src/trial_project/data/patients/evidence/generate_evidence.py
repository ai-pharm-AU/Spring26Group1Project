import json
import pandas as pd
from pandas import DataFrame
from trial_project.api import generate_client
from trial_project.context import data_dir
from trial_project.data.patients.load_patient import get_patient_llm_json
from trial_project.data.patients.evidence.schema import PatientEvidence

evidence_path = data_dir / "processed_data" / "patient_evidence.parquet"

_EVIDENCE_COLUMNS = ["patient_id", "patient_evidence"]


def _empty_evidence_frame() -> DataFrame:
  return DataFrame(columns=_EVIDENCE_COLUMNS)


def load_all_patient_evidence() -> DataFrame:
  """Load all cached patient evidence rows."""
  if not evidence_path.exists():
    return _empty_evidence_frame()

  evidence_df = pd.read_parquet(evidence_path)
  for col in _EVIDENCE_COLUMNS:
    if col not in evidence_df.columns:
      evidence_df[col] = pd.NA
  return evidence_df


def load_patient_evidence(patient_id: str, evidence_df: DataFrame | None = None) -> str | None:
  """Load cached evidence text for one patient."""
  if evidence_df is None:
    evidence_df = load_all_patient_evidence()

  patient_rows = evidence_df[evidence_df["patient_id"] == patient_id]
  if patient_rows.empty:
    return None

  evidence_values = patient_rows["patient_evidence"].dropna()
  if evidence_values.empty:
    return None

  return str(evidence_values.iloc[0])


def save_patient_evidence(patient_id: str, patient_evidence: str) -> None:
  """Upsert one patient evidence row into the parquet cache."""
  evidence_df = load_all_patient_evidence()

  for col in _EVIDENCE_COLUMNS:
    if col not in evidence_df.columns:
      evidence_df[col] = pd.NA

  mask = evidence_df["patient_id"] == patient_id
  new_row = {
    "patient_id": patient_id,
    "patient_evidence": patient_evidence,
  }

  evidence_df = evidence_df.loc[~mask]
  evidence_df = pd.concat([evidence_df, DataFrame([new_row])], ignore_index=True)

  evidence_path.parent.mkdir(parents=True, exist_ok=True)
  evidence_df.to_parquet(evidence_path, index=False)


patient_evidence_prompt = """You are extracting a reusable patient evidence profile from one Synthea patient record.

Goal:

Convert one patient record into a structured patient evidence index that can later be matched against many trial criteria.

Important rules:
- The patient record is the only source of patient evidence.
- Do not infer diagnoses, performance status, biomarkers, pregnancy status, or treatment response unless explicitly supported.
- Preserve dates, units, values, and start/stop timing exactly.
- Keep diagnosis history, medication history, procedures, labs, vitals, demographics, and missingness separate.
- Extract reusable evidence once so it can support many future trial matches.
- Prefer normalized clinical concepts and original evidence text.
- Do not determine eligibility here.

Evidence coverage notes:
- Often available: age derivable from birthdate, sex, race, ethnicity, conditions, diagnosis timing, medications, BMI, weight, blood pressure, creatinine, some labs, procedures.
- Often sparse or missing: ECOG, Karnofsky, QTc, histology/subtype, molecular markers, genetic syndromes, NYHA, LVEF, Child-Pugh, detailed prior response/failure.

Input patient record schema:
- patient: Id, BIRTHDATE, GENDER, RACE, ETHNICITY
- encounters: START, STOP, ENCOUNTERCLASS, DESCRIPTION
- conditions: START, STOP, DESCRIPTION
- medications: START, STOP, DESCRIPTION
- observations: DATE, DESCRIPTION, VALUE, UNITS
- procedures
- allergies
- immunizations
- careplans

Instructions:
1. Read the full patient record.
2. Normalize demographics into reusable fields.
3. Extract active and historical conditions with dates.
4. Extract medication exposure history with dates.
5. Extract procedures with dates.
6. Extract clinically relevant observations with values, units, and dates.
7. Group observations into useful categories such as body size, blood pressure, renal labs, hepatic labs, hematology, metabolic labs, smoking, and other patient-reported measures.
8. Preserve uncertainty and missing fields explicitly.
9. Create a reusable patient evidence profile that can be matched to many trials.
10. Return JSON only.

Use the provided JSON schema for output shape and constraints.

Input:

"""


def generate_patient_evidence(patient_id: str) -> str:
  """Generate patient evidence JSON for one patient."""
  client = generate_client()
  patient_record_json = get_patient_llm_json(patient_id)

  response = client.responses.parse(
    model="gpt-5-mini",
    instructions=patient_evidence_prompt,
    input=patient_record_json,
    text_format=PatientEvidence
  )
  validated = PatientEvidence.model_validate_json(response.output_text)
  return json.dumps(validated.model_dump(), ensure_ascii=True)


def get_patient_evidence(patient_id: str) -> str:
  """Load cached patient evidence or generate and save it."""
  evidence_df = load_all_patient_evidence()
  existing = load_patient_evidence(patient_id, evidence_df)
  if existing is not None:
    return existing

  patient_evidence = generate_patient_evidence(patient_id)
  save_patient_evidence(patient_id, patient_evidence)
  return patient_evidence


