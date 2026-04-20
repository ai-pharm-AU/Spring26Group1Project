import argparse
from trial_project.data.trials.load import load_trial_json_llm, load_trial_ids
import pandas as pd
from pandas import DataFrame
from trial_project.api import generate_client
from trial_project.context import data_dir

eligibility_path = data_dir / "processed_data" / "trials" / "eligibility.parquet"

_ELIGIBILITY_COLUMNS = [
  "trial_id",
  "model_name",
  "eligibility_criteria",
  "eligibility_criteria_llm",
]


def _normalize_model_name(model_name: str | None) -> str:
  if model_name is None or pd.isna(model_name):
    return ""
  return str(model_name).strip()


def _empty_eligibility_frame() -> DataFrame:
  return DataFrame(columns=_ELIGIBILITY_COLUMNS)


def _ensure_eligibility_schema(eligibility_df: DataFrame) -> DataFrame:
  for col in _ELIGIBILITY_COLUMNS:
    if col not in eligibility_df.columns:
      eligibility_df[col] = pd.NA

  # Legacy rows were written before model_name existed.
  model_series = eligibility_df["model_name"].astype("string")
  model_series = model_series.fillna("").str.strip()
  eligibility_df["model_name"] = model_series.replace("", "gpt-5-mini")
  return eligibility_df


def _load_all_eligibility() -> DataFrame:
  if not eligibility_path.exists():
    return _empty_eligibility_frame()
  eligibility_df = pd.read_parquet(eligibility_path)
  return _ensure_eligibility_schema(eligibility_df)


def backfill_eligibility_model_name(default_model_name: str = "gpt-5-mini") -> int:
  """Backfill missing/blank model_name values and persist the parquet cache."""
  if not eligibility_path.exists():
    return 0

  eligibility_df = pd.read_parquet(eligibility_path)
  before_count = len(eligibility_df)

  for col in _ELIGIBILITY_COLUMNS:
    if col not in eligibility_df.columns:
      eligibility_df[col] = pd.NA

  model_series = eligibility_df["model_name"].astype("string")
  model_series = model_series.fillna("").str.strip()
  eligibility_df["model_name"] = model_series.replace("", default_model_name)

  eligibility_path.parent.mkdir(parents=True, exist_ok=True)
  eligibility_df.to_parquet(eligibility_path, index=False)
  return before_count

def load_trial_eligibility_criteria(trial_id) -> DataFrame:
  # return the eligibility criteria for the trial
  eligibility_df = _load_all_eligibility()
  return eligibility_df[eligibility_df["trial_id"] == trial_id]

def get_trial_eligibility_text(trial_id) -> str:
  # return the eligibility criteria text for the trial for the llm
  eligibility_df = load_trial_eligibility_criteria(trial_id)
  if eligibility_df.empty:
    return "No eligibility criteria found for this trial."
  criteria = eligibility_df["eligibility_criteria"].dropna()
  if criteria.empty:
    return "No eligibility criteria found for this trial."
  return str(criteria.iloc[0])


# def get_trial_eligibility_llm(trial_id) -> str:
#   # return the eligibility criteria summary made by the llm
#   eligibility_df = load_trial_eligibility_criteria(trial_id)
#   if eligibility_df.empty:
#     return "No eligibility criteria found for this trial."
#   criteria = eligibility_df["eligibility_criteria_llm"]
#   criteria_text = criteria.str() # idk if right
#   return criteria_text

def get_trial_eligibility_llm(trial_id, model_name: str = "gpt-5-mini") -> str:
  # try to load from df, if not there, generate with llm and save to df
  eligibility_df = _load_all_eligibility()
  model_key = _normalize_model_name(model_name)
  existing_model_key = eligibility_df["model_name"].apply(_normalize_model_name)
  trial_mask = (
    (eligibility_df["trial_id"] == trial_id)
    & (existing_model_key == model_key)
  )
  existing = eligibility_df.loc[trial_mask, "eligibility_criteria_llm"].dropna()
  if not existing.empty:
    return str(existing.iloc[0])

  eligibility_text = create_eligibility_llm(trial_id, model_name=model_key)

  new_row = {
    "trial_id": trial_id,
    "model_name": model_key,
    "eligibility_criteria": pd.NA,
    "eligibility_criteria_llm": eligibility_text,
  }
  eligibility_df = eligibility_df.loc[~trial_mask]
  eligibility_df = pd.concat([eligibility_df, DataFrame([new_row])], ignore_index=True)

  eligibility_path.parent.mkdir(parents=True, exist_ok=True)
  eligibility_df.to_parquet(eligibility_path, index=False)
  return eligibility_text

def create_eligibility_llm(trial_id, model_name: str = "gpt-5-mini"):
  client = generate_client()
  trial_info = load_trial_json_llm(trial_id)
  prompt = """You are extracting normalized eligibility criteria from one clinical trial record.
Goal: 

Convert one trial record into atomic eligibility criteria that can later be matched against normalized patient evidence. 

Important rules: 
- The trial record is the source of truth for eligibility requirements. 
- Read inclusion_criteria and exclusion_criteria together. 
- Some exclusion language may appear inside inclusion_criteria. 
- Do not create new eligibility requirements from brief_summary alone. 
- Use brief_summary, diseases_list, and drugs_list only to clarify ambiguous wording. 
- Preserve inclusion vs exclusion polarity exactly. 
- Preserve age ranges, sex restrictions, disease requirements, prior therapy requirements, timing windows, and numeric thresholds exactly. 
- If the record describes multiple populations, substudies, or arms, preserve that ambiguity rather than collapsing them into one forced rule. 
- Do not infer missing criteria. 
- Keep criteria atomic: one requirement per item. 

FDA guidelines: 
- Preserve criterion meaning. 
- Prefer specific measurable concepts over broader paraphrases. 
- Do not broaden or narrow the requirements. 
- If a criterion is unusually broad, still encode it faithfully. 
- Keep unresolved ambiguity in notes. 

Input trial record schema: 
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

Instructions: 
1. Read the full trial record. 
2. Parse raw eligibility text from both inclusion_criteria and exclusion_criteria. 
3. Split all requirements into atomic criteria. 
4. Assign each atomic criterion a criterion_type of inclusion or exclusion. 
5. Normalize each criterion into structured concepts. 
6. Capture temporal rules, numeric rules, and subgroup restrictions explicitly. 
7. Add ambiguity notes when the text is incomplete, mixed, or arm-specific. 
8. Return JSON only. 

Return JSON in this exact shape: 
{ 

  "trial_id": "", 
  "brief_title": "", 
  "phase": "", 
  "trial_context": { 
    "diseases_list": [], 
    "drugs_list": [], 
    "brief_summary_short": "" 
  }, 

  "parsing_notes": { 
    "format_issues": [], 
    "mixed_polarity_issues": [], 
    "substudy_or_arm_issues": [], 
    "other_ambiguities": []
  },

  "atomic_criteria": [ 
    { 
      "criterion_id": "", 
      "criterion_type": "inclusion|exclusion", 
      "criterion_text": "", 
      "normalized_requirement": "", 
      "concept_groups": { 
        "disease_or_condition": [], 
        "severity_or_stage": [], 
        "symptoms_signs": [], 
        "demographics": [], 
        "organ_function": [], 
        "comorbidities": [], 
        "prior_therapy": [], 
        "concomitant_medications": [], 
        "biomarkers_genetics": [], 
        "pregnancy_lactation": [], 
        "physiologic_parameters": [], 
        "procedures": [], 
        "logistics_or_followup": [], 
        "temporal_constraints": [], 
        "numeric_constraints": [] 
      }, 

      "synonyms": [], 
      "structured_terms": [], 
      "required_patient_evidence": [], 
      "low_matchability_fields": [], 
      "notes": [] 
    } 
  ] 
}
"""

  response = client.responses.create(
    model=model_name,
    instructions=prompt,
    input=trial_info,
)
  return response.output_text


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Generate trial eligibility criteria and cache by model.",
  )
  parser.add_argument(
    "--model-name",
    default="gpt-5-mini",
    help="LLM model name used to generate trial eligibility criteria.",
  )
  parser.add_argument(
    "--backfill-only",
    action="store_true",
    help="Backfill model_name in existing eligibility cache and exit.",
  )
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  migrated_rows = backfill_eligibility_model_name(default_model_name="gpt-5-mini")
  if migrated_rows > 0:
    print(f"Backfilled eligibility cache model_name for {migrated_rows} rows.")
  if args.backfill_only:
    raise SystemExit(0)

  trial_ids = load_trial_ids()
  for trial_id in trial_ids:
    print(f"Processing trial {trial_id}")
    eligibility_text = get_trial_eligibility_llm(trial_id, model_name=args.model_name)
    print(f"Eligibility criteria for trial {trial_id}: {eligibility_text}")
