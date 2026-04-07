import json

import pandas as pd
from pandas import DataFrame

from trial_project.api import generate_client
from trial_project.context import data_dir
from trial_project.data.patients.load_patient import get_patient_llm_json

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


def _patient_evidence_prompt() -> str:
  return """You are extracting a reusable patient evidence profile from one Synthea patient record.

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

Return JSON in this exact shape:

{
  "patient_id": "",
  "demographics": {
    "birthdate": "",
    "sex": "",
    "race": "",
    "ethnicity": "",
    "derived_age_if_needed_at_match_time": true
  },

  "condition_index": [
    {
      "normalized_condition": "",
      "original_text": "",
      "start_date": "",
      "end_date": "",
      "status": "active|historical|unknown",
      "synonyms": []
    }
  ],

  "medication_index": [
    {
      "normalized_medication": "",
      "original_text": "",
      "start_date": "",
      "end_date": "",
      "status": "current|past|unknown",
      "drug_class_if_clear": "",
      "synonyms": []
    }
  ],

  "procedure_index": [
    {
      "normalized_procedure": "",
      "original_text": "",
      "date_or_start": "",
      "end_date": "",
      "synonyms": []
    }
  ],

  "observation_index": [
    {
      "category": "body_size|blood_pressure|renal|hepatic|hematology|metabolic|cardiac|smoking|social|other",
      "normalized_name": "",
      "original_text": "",
      "value": "",
      "units": "",
      "date": "",
      "interpretation_if_explicit": ""
    }
  ],

  "encounter_index": [
    {
      "encounter_class": "",
      "description": "",
      "start_date": "",
      "end_date": ""
    }
  ],

  "evidence_flags": {
    "has_performance_status": false,
    "has_qtc": false,
    "has_histology": false,
    "has_biomarkers": false,
    "has_nyha": false,
    "has_lvef": false,
    "has_child_pugh": false,
    "has_pregnancy_lactation_evidence": false
  },

  "missingness_notes": [],
  "patient_summary": {
    "major_conditions": [],
    "major_medications": [],
    "major_recent_labs_or_vitals": [],
    "important_unknowns": []
  }
}

Input:

{{patient_record_json}}
"""


def generate_patient_evidence(patient_id: str) -> str:
  """Generate patient evidence JSON for one patient."""
  client = generate_client()
  patient_record_json = get_patient_llm_json(patient_id)

  response = client.responses.create(
    model="gpt-5-mini",
    instructions=_patient_evidence_prompt(),
    input=patient_record_json,
  )
  return response.output_text


def get_patient_evidence(patient_id: str) -> str:
  """Load cached patient evidence or generate and save it."""
  evidence_df = load_all_patient_evidence()
  existing = load_patient_evidence(patient_id, evidence_df)
  if existing is not None:
    return existing

  patient_evidence = generate_patient_evidence(patient_id)
  try:
    normalized_evidence = json.dumps(json.loads(patient_evidence), ensure_ascii=True)
  except json.JSONDecodeError:
    normalized_evidence = patient_evidence

  save_patient_evidence(patient_id, normalized_evidence)
  return normalized_evidence


if __name__ == "__main__":
  patients_path = data_dir / "processed_data" / "patients.parquet"
  patients_df = pd.read_parquet(patients_path, columns=["Id"])

  for _, patient_row in patients_df.iterrows():
    patient_id = patient_row["Id"]
    print(f"Processing patient {patient_id}")
    patient_evidence = get_patient_evidence(patient_id)
    print(f"Patient evidence for {patient_id}: {patient_evidence}")
