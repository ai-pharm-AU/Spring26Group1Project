import argparse
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field

from trial_project.api import generate_client
from trial_project.context import data_dir
from trial_project.data.trials.eligibility import get_trial_eligibility_llm
from trial_project.data.trials.load import load_trial_json_llm
from trial_project.labeling.pairs import load_matched_pairs

verification_path = data_dir / "processed_data" / "trials" / "eligibility_verification.parquet"

_VERIFICATION_COLUMNS = [
   "trial_id",
   "trial_eligibility_verification",
   "corrected_trial_eligibility",
]


class RawTrialEvidence(BaseModel):
   model_config = ConfigDict(extra="forbid")

   source_section: Literal[
      "id",
      "brief_title",
      "phase",
      "drugs",
      "drugs_list",
      "diseases",
      "diseases_list",
      "enrollment",
      "inclusion_criteria",
      "exclusion_criteria",
      "brief_summary",
   ] = "id"
   original_text: str = ""
   date_or_start: str = ""
   end_date: str = ""
   value: str = ""
   units: str = ""


class TrialClaimReview(BaseModel):
   model_config = ConfigDict(extra="forbid")

   claim_id: str = ""
   criterion_id: str = ""
   original_claim: str = ""
   claim_type: Literal[
      "trial_metadata",
      "condition",
      "drug_or_intervention",
      "eligibility_criterion",
      "phase",
      "enrollment",
      "summary",
      "other",
   ] = "other"
   verification_status: Literal[
      "supported",
      "partially_supported",
      "contradicted",
      "not_found",
   ] = "not_found"
   verification_explanation: str = ""
   matched_raw_evidence: list[RawTrialEvidence] = Field(default_factory=list)
   corrected_claim: str = ""
   is_proxy_only: bool = False
   should_be_used_for_eligibility: bool = True
   confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class OmittedRelevantTrialEvidence(BaseModel):
   model_config = ConfigDict(extra="forbid")

   criterion_id: str = ""
   why_relevant: str = ""
   raw_evidence: list[RawTrialEvidence] = Field(default_factory=list)
   suggested_claim: str = ""


class ContradictoryTrialEvidence(BaseModel):
   model_config = ConfigDict(extra="forbid")

   claim_id: str = ""
   description: str = ""
   raw_evidence: list[RawTrialEvidence] = Field(default_factory=list)


class TrialContext(BaseModel):
   model_config = ConfigDict(extra="forbid")

   diseases_list: list[str] = Field(default_factory=list)
   drugs_list: list[str] = Field(default_factory=list)
   brief_summary_short: str = ""


class ParsingNotes(BaseModel):
   model_config = ConfigDict(extra="forbid")

   format_issues: list[str] = Field(default_factory=list)
   mixed_polarity_issues: list[str] = Field(default_factory=list)
   substudy_or_arm_issues: list[str] = Field(default_factory=list)
   other_ambiguities: list[str] = Field(default_factory=list)


class ConceptGroups(BaseModel):
   model_config = ConfigDict(extra="forbid")

   disease_or_condition: list[str] = Field(default_factory=list)
   severity_or_stage: list[str] = Field(default_factory=list)
   symptoms_signs: list[str] = Field(default_factory=list)
   demographics: list[str] = Field(default_factory=list)
   organ_function: list[str] = Field(default_factory=list)
   comorbidities: list[str] = Field(default_factory=list)
   prior_therapy: list[str] = Field(default_factory=list)
   concomitant_medications: list[str] = Field(default_factory=list)
   biomarkers_genetics: list[str] = Field(default_factory=list)
   pregnancy_lactation: list[str] = Field(default_factory=list)
   physiologic_parameters: list[str] = Field(default_factory=list)
   procedures: list[str] = Field(default_factory=list)
   logistics_or_followup: list[str] = Field(default_factory=list)
   temporal_constraints: list[str] = Field(default_factory=list)
   numeric_constraints: list[str] = Field(default_factory=list)


class AtomicCriterion(BaseModel):
   model_config = ConfigDict(extra="forbid")

   criterion_id: str = ""
   criterion_type: Literal["inclusion", "exclusion"] = "inclusion"
   criterion_text: str = ""
   normalized_requirement: str = ""
   concept_groups: ConceptGroups = Field(default_factory=ConceptGroups)
   synonyms: list[str] = Field(default_factory=list)
   structured_terms: list[str] = Field(default_factory=list)
   required_patient_evidence: list[str] = Field(default_factory=list)
   low_matchability_fields: list[str] = Field(default_factory=list)
   notes: list[str] = Field(default_factory=list)


class CorrectedTrialEligibility(BaseModel):
   model_config = ConfigDict(extra="forbid")

   trial_id: str = ""
   brief_title: str = ""
   phase: str = ""
   trial_context: TrialContext = Field(default_factory=TrialContext)
   parsing_notes: ParsingNotes = Field(default_factory=ParsingNotes)
   atomic_criteria: list[AtomicCriterion] = Field(default_factory=list)


class TrialEligibilityVerification(BaseModel):
   model_config = ConfigDict(extra="forbid")

   trial_id: str = ""
   overall_verification_status: Literal[
      "fully_verified",
      "partially_verified",
      "verification_failed",
      "manual_review_needed",
   ] = "manual_review_needed"
   overall_verification_summary: str = ""
   fit_for_trial_eligibility_review: bool = False
   fit_for_trial_eligibility_review_explanation: str = ""
   corrected_trial_eligibility: CorrectedTrialEligibility = Field(default_factory=CorrectedTrialEligibility)
   claim_reviews: list[TrialClaimReview] = Field(default_factory=list)
   omitted_relevant_evidence: list[OmittedRelevantTrialEvidence] = Field(default_factory=list)
   unsupported_or_problematic_claims: list[str] = Field(default_factory=list)
   contradictory_trial_evidence: list[ContradictoryTrialEvidence] = Field(default_factory=list)
   manual_review_flags: list[str] = Field(default_factory=list)
   verification_notes: list[str] = Field(default_factory=list)


def _empty_verification_frame() -> DataFrame:
   return DataFrame(columns=_VERIFICATION_COLUMNS)


def load_all_trial_eligibility_verification() -> DataFrame:
   """Load all cached trial eligibility verification rows."""
   if not verification_path.exists():
      return _empty_verification_frame()

   verification_df = pd.read_parquet(verification_path)
   for col in _VERIFICATION_COLUMNS:
      if col not in verification_df.columns:
         verification_df[col] = pd.NA
   return verification_df


def load_trial_eligibility_verification(
   trial_id: str,
   verification_df: DataFrame | None = None,
) -> str | None:
   """Load cached corrected trial eligibility JSON text for one trial."""
   if verification_df is None:
      verification_df = load_all_trial_eligibility_verification()

   trial_rows = verification_df[verification_df["trial_id"] == trial_id]
   if trial_rows.empty:
      return None

   corrected_values = trial_rows["corrected_trial_eligibility"].dropna()
   if corrected_values.empty:
      return None

   return str(corrected_values.iloc[0])


def save_trial_eligibility_verification(
   trial_id: str,
   trial_eligibility_verification: str,
   corrected_trial_eligibility: str,
) -> None:
   """Upsert one trial eligibility verification row into the parquet cache."""
   verification_df = load_all_trial_eligibility_verification()

   for col in _VERIFICATION_COLUMNS:
      if col not in verification_df.columns:
         verification_df[col] = pd.NA

   mask = verification_df["trial_id"] == trial_id
   new_row = {
      "trial_id": trial_id,
      "trial_eligibility_verification": trial_eligibility_verification,
      "corrected_trial_eligibility": corrected_trial_eligibility,
   }

   verification_df = verification_df.loc[~mask]
   verification_df = pd.concat([verification_df, DataFrame([new_row])], ignore_index=True)

   verification_path.parent.mkdir(parents=True, exist_ok=True)
   verification_df.to_parquet(verification_path, index=False)


def _parse_json_or_raw(text: str) -> dict[str, Any] | list[Any] | str:
   try:
      return json.loads(text)
   except json.JSONDecodeError:
      return text


def _trial_eligibility_verification_prompt() -> str:
   return """You are verifying the accuracy of one trial's extracted eligibility information against the original trial record.

Goal:

Verify whether a single trial's previously extracted evidence is accurate to the raw trial record and whether the extracted evidence is fit to be used for trial eligibility review.

Important rules:

- The raw trial record is the only source of truth for trial facts.
- Do not invent evidence.
- Do not treat missing evidence as negative evidence.
- Preserve dates, units, inclusion vs exclusion status, negation, temporality, and criterion polarity exactly.
- If extracted evidence is partly correct but overstated, mark it as partially_supported and explain the overstatement.
- If extracted evidence is absent from the trial record, mark it as not_found.
- If the trial record contains contradictory evidence, mark it as contradicted.

Trial record sections may include:
- id
- brief_title
- phase
- drugs
- drugs_list
- diseases
- diseases_list
- enrollment
- inclusion_criteria
- exclusion_criteria
- brief_summary

For each extracted evidence claim, verify when relevant:
- Concept accuracy
- Trial metadata accuracy
- Diagnosis or condition accuracy
- Drug or intervention accuracy
- Eligibility criterion accuracy
- Units accuracy
- Date accuracy
- Temporal interpretation accuracy
- Inclusion vs exclusion status
- Negation or absence claims
- Whether the claim is direct evidence or only a proxy

Verification labels:
- supported = clearly and accurately supported by the raw trial record
- partially_supported = related evidence exists, but the extracted claim is incomplete, overstated, underspecified, or imprecise
- contradicted = the raw trial record contains evidence against the extracted claim
- not_found = no supporting evidence for the claim was found in the raw trial record

Instructions:
1. Read the extracted trial evidence under review.
2. Read the raw trial record and treat it as the source of truth.
3. Verify each extracted evidence item against the raw trial record.
4. For each item, return:
    - verification_status
    - a short explanation
    - the best matching raw record evidence, if any
    - a corrected version of the claim when the original is inaccurate or overstated
5. Build corrected_trial_eligibility as a complete corrected trial eligibility object suitable for downstream matching.
6. If extracted evidence omits an important trial fact clearly present in the raw record and relevant to eligibility, add it under omitted_relevant_evidence.
7. If extracted evidence contains a claim that should not be used for eligibility because it is only a weak proxy, mark is_proxy_only and should_be_used_for_eligibility accordingly.
8. If an age requirement or other eligibility restriction must be verified, verify it against the raw trial record exactly as written.
9. Keep the output compact and evidence-based.
10. Return JSON only.

Use the provided JSON schema for output shape and constraints.

Input:

{{verification_input_json}}
"""


def _verification_input(
   trial_id: str,
   trial_eligibility: str,
   trial_record_json: str,
) -> str:
   payload = {
      "trial_id": trial_id,
      "trial_eligibility_under_review": _parse_json_or_raw(trial_eligibility),
      "raw_trial_record": _parse_json_or_raw(trial_record_json),
   }
   return json.dumps(payload, ensure_ascii=True)


def generate_trial_eligibility_verification(
   trial_id: str,
   model_name: str = "gpt-5-mini",
) -> tuple[str, str]:
   """Generate verification JSON and corrected trial eligibility JSON for one trial."""
   client = generate_client()
   trial_eligibility = get_trial_eligibility_llm(trial_id)
   trial_record_json = load_trial_json_llm(trial_id)
   verification_input = _verification_input(
      trial_id=trial_id,
      trial_eligibility=trial_eligibility,
      trial_record_json=trial_record_json,
   )

   response = client.responses.parse(
      model=model_name,
      instructions=_trial_eligibility_verification_prompt(),
      input=verification_input,
      text_format=TrialEligibilityVerification,
   )
   validated = TrialEligibilityVerification.model_validate_json(response.output_text)
   verification_json = json.dumps(validated.model_dump(), ensure_ascii=True)
   corrected_json = json.dumps(validated.corrected_trial_eligibility.model_dump(), ensure_ascii=True)
   return verification_json, corrected_json


def get_trial_eligibility_verification(
   trial_id: str,
   model_name: str = "gpt-5-mini",
   use_cache: bool = True,
) -> str:
   """Return corrected trial eligibility JSON text (cached when available)."""
   if use_cache:
      verification_df = load_all_trial_eligibility_verification()
      existing = load_trial_eligibility_verification(trial_id, verification_df)
      if existing is not None:
         return existing

   verification, corrected = generate_trial_eligibility_verification(
      trial_id=trial_id,
      model_name=model_name,
   )
   save_trial_eligibility_verification(
      trial_id=trial_id,
      trial_eligibility_verification=verification,
      corrected_trial_eligibility=corrected,
   )
   return corrected


def _load_matched_trial_ids(
   matched_pairs_source: str | Path | None = None,
) -> list[str]:
   try:
      matched_pairs = load_matched_pairs(matched_pairs_source)
   except FileNotFoundError:
      return []

   if matched_pairs.empty:
      return []

   trial_ids = matched_pairs["trial_id"].dropna().astype(str).str.strip()
   trial_ids = trial_ids[trial_ids != ""]
   return sorted(trial_ids.unique().tolist())


def generate_and_verify_all_trials(
   model_name: str = "gpt-5-mini",
   use_cache: bool = True,
   continue_on_error: bool = False,
   matched_pairs_source: str | Path | None = None,
) -> tuple[int, int]:
   """Generate and verify trial eligibility evidence for matched trial IDs only."""
   trial_ids = _load_matched_trial_ids(matched_pairs_source)
   if not trial_ids:
      print("No matched trial pairs found; skipping trial eligibility verification generation.")
      return 0, 0

   total = len(trial_ids)

   succeeded = 0
   failed = 0

   for idx, trial_id in enumerate(trial_ids, start=1):
      print(f"[{idx}/{total}] Processing trial {trial_id}")
      try:
         get_trial_eligibility_verification(
            trial_id=trial_id,
            model_name=model_name,
            use_cache=use_cache,
         )
         succeeded += 1
      except Exception as exc:
         failed += 1
         print(f"[{idx}/{total}] Failed trial {trial_id}: {exc}")
         if not continue_on_error:
            raise

   print(f"Completed trial eligibility verification: success={succeeded}, failed={failed}")
   return succeeded, failed


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(
      description="Generate and verify eligibility extraction for matched trials.",
   )
   parser.add_argument(
      "--model-name",
      default="gpt-5-mini",
      help="LLM model name used for verification.",
   )
   parser.add_argument(
      "--no-cache",
      action="store_true",
      help="Disable verification cache and re-run verification for each trial.",
   )
   parser.add_argument(
      "--continue-on-error",
      action="store_true",
      help="Continue processing remaining trials when one fails.",
   )
   parser.add_argument(
      "--matched-pairs-file",
      default=None,
      help="Matched patient-trial parquet used to select trial IDs.",
   )
   return parser.parse_args()


def main() -> int:
   args = parse_args()
   _, failed = generate_and_verify_all_trials(
      model_name=args.model_name,
      use_cache=not args.no_cache,
      continue_on_error=args.continue_on_error,
      matched_pairs_source=args.matched_pairs_file,
   )
   return 0 if failed == 0 else 1


if __name__ == "__main__":
   raise SystemExit(main())

