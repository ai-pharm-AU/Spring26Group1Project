from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class Demographics(BaseModel):
  model_config = ConfigDict(extra="forbid")

  birthdate: str = ""
  sex: str = ""
  race: str = ""
  ethnicity: str = ""
  age: str = ""


class ConditionIndexItem(BaseModel):
  model_config = ConfigDict(extra="forbid")

  normalized_condition: str = ""
  original_text: str = ""
  start_date: str = ""
  end_date: str = ""
  status: Literal["active", "historical", "unknown"] = "unknown"
  synonyms: list[str] = Field(default_factory=list)


class MedicationIndexItem(BaseModel):
  model_config = ConfigDict(extra="forbid")

  normalized_medication: str = ""
  original_text: str = ""
  start_date: str = ""
  end_date: str = ""
  status: Literal["current", "past", "unknown"] = "unknown"
  drug_class_if_clear: str = ""
  synonyms: list[str] = Field(default_factory=list)


class ProcedureIndexItem(BaseModel):
  model_config = ConfigDict(extra="forbid")

  normalized_procedure: str = ""
  original_text: str = ""
  date_or_start: str = ""
  end_date: str = ""
  synonyms: list[str] = Field(default_factory=list)


class ObservationIndexItem(BaseModel):
  model_config = ConfigDict(extra="forbid")

  category: Literal[
    "body_size",
    "blood_pressure",
    "renal",
    "hepatic",
    "hematology",
    "metabolic",
    "cardiac",
    "smoking",
    "social",
    "other",
  ] = "other"
  normalized_name: str = ""
  original_text: str = ""
  value: str = ""
  units: str = ""
  date: str = ""
  interpretation_if_explicit: str = ""


class EncounterIndexItem(BaseModel):
  model_config = ConfigDict(extra="forbid")

  encounter_class: str = ""
  description: str = ""
  start_date: str = ""
  end_date: str = ""


class EvidenceFlags(BaseModel):
  model_config = ConfigDict(extra="forbid")

  has_performance_status: bool = False
  has_qtc: bool = False
  has_histology: bool = False
  has_biomarkers: bool = False
  has_nyha: bool = False
  has_lvef: bool = False
  has_child_pugh: bool = False
  has_pregnancy_lactation_evidence: bool = False


class PatientSummary(BaseModel):
  model_config = ConfigDict(extra="forbid")
  major_conditions: list[str] = Field(default_factory=list)
  major_medications: list[str] = Field(default_factory=list)
  major_recent_labs_or_vitals: list[str] = Field(default_factory=list)
  important_unknowns: list[str] = Field(default_factory=list)


class PatientEvidence(BaseModel):
  model_config = ConfigDict(extra="forbid")

  patient_id: str = ""
  demographics: Demographics
  condition_index: list[ConditionIndexItem] = Field(default_factory=list)
  medication_index: list[MedicationIndexItem] = Field(default_factory=list)
  procedure_index: list[ProcedureIndexItem] = Field(default_factory=list)
  observation_index: list[ObservationIndexItem] = Field(default_factory=list)
  encounter_index: list[EncounterIndexItem] = Field(default_factory=list)
  evidence_flags: EvidenceFlags
  missingness_notes: list[str] = Field(default_factory=list)
  patient_summary: PatientSummary
