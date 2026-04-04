from trial_project.data.trials.load import get_trial
import pandas as pd
from pandas import DataFrame
from trial_project.context import data_dir

eligibility_path = data_dir / "processed_data" / "trials" / "eligibility.parquet"

def load_trial_eligibility_criteria(trial_id) -> DataFrame:
  # return the eligibility criteria text for the trial for the llm
  eligibility_df = pd.read_parquet(eligibility_path)
  return eligibility_df[eligibility_df["trial_id"] == trial_id]

def get_trial_eligibility_text(trial_id) -> str:
  # return the eligibility criteria text for the trial for the llm
  eligibility_df = load_trial_eligibility_criteria(trial_id)
  if eligibility_df.empty:
    return "No eligibility criteria found for this trial."
  criteria = eligibility_df["eligibility_criteria"]
  criteria_text = criteria.str() # idk if right
  return criteria_text