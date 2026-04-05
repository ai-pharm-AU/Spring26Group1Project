"""
load trials
"""

import pandas as pd
from trial_project.context import data_dir

trials_path = data_dir / "processed_data" / "trials" / "trials.parquet"

def load_trial_ids() -> list:
  trial_df = load_all_trials()
  return trial_df["Id"].tolist()

def load_trial(trial_id: str) -> pd.DataFrame:
  trial_df = load_all_trials()
  return trial_df[trial_df["Id"] == trial_id]

def load_all_trials() -> pd.DataFrame:
  return pd.read_parquet(trials_path)

# TODO trial json processed for llm
def load_trial_json_llm(trial_id: str) -> str:
  return get_trial_json(trial_id)

def get_trial_json(trial_id: str) -> dict:
  # Load one trial and all related records as a JSON-serializable dict
  output = {id: "trial_1", "trial": {"name": "Trial 1", "description": "This is a trial about testing.", "eligibility_criteria": "Must be over 18."}}
  return output
  # trial_df = load_all_trials()
  # trial_rows = trial_df[trial_df["Id"] == trial_id]
  # if trial_rows.empty:
  #   raise ValueError(f"Trial not found: {trial_id}")
	
  # trial_data = trial_rows.iloc[0].dropna().to_dict()
  # output = {"id": trial_id, "trial": trial_data}
  # return output