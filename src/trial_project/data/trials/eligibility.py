from trial_project.data.trials.load import load_trial_json_llm, load_trial_ids
import pandas as pd
from pandas import DataFrame
from trial_project.api import generate_client
from trial_project.context import data_dir

eligibility_path = data_dir / "processed_data" / "trials" / "eligibility.parquet"

def load_trial_eligibility_criteria(trial_id) -> DataFrame:
  # return the eligibility criteria for the trial
  if not eligibility_path.exists():
    return DataFrame(columns=["trial_id", "eligibility_criteria", "eligibility_criteria_llm"])
  eligibility_df = pd.read_parquet(eligibility_path)
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

def get_trial_eligibility_llm(trial_id) -> str:
  # try to load from df, if not there, generate with llm and save to df
  if eligibility_path.exists():
    eligibility_df = pd.read_parquet(eligibility_path)
  else:
    eligibility_df = DataFrame(columns=["trial_id", "eligibility_criteria", "eligibility_criteria_llm"])

  # Keep schema stable when reading older files.
  for col in ["trial_id", "eligibility_criteria", "eligibility_criteria_llm"]:
    if col not in eligibility_df.columns:
      eligibility_df[col] = pd.NA

  trial_mask = eligibility_df["trial_id"] == trial_id
  existing = eligibility_df.loc[trial_mask, "eligibility_criteria_llm"].dropna()
  if not existing.empty:
    return str(existing.iloc[0])

  eligibility_text = create_eligibility_llm(trial_id)

  if trial_mask.any():
    eligibility_df.loc[trial_mask, "eligibility_criteria_llm"] = eligibility_text
  else:
    new_row = {
      "trial_id": trial_id,
      "eligibility_criteria": pd.NA,
      "eligibility_criteria_llm": eligibility_text,
    }
    eligibility_df = pd.concat([eligibility_df, DataFrame([new_row])], ignore_index=True)

  eligibility_path.parent.mkdir(parents=True, exist_ok=True)
  eligibility_df.to_parquet(eligibility_path, index=False)
  return eligibility_text

def create_eligibility_llm(trial_id):
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
    model="gpt-5-mini",
    instructions=prompt,
    input=trial_info,
)
  return response.output_text

if __name__ == "__main__":
  trial_ids = load_trial_ids()
  for trial_id in trial_ids:
    print(f"Processing trial {trial_id}")
    eligibility_text = get_trial_eligibility_llm(trial_id)
    print(f"Eligibility criteria for trial {trial_id}: {eligibility_text}")
