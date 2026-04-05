"""
get the relevant trials for a patient based on keywords
"""

import pandas as pd
from trial_project.context import results_dir
from trial_project.retrieval.hybrid_fusion import rank_trials_for_conditions
from trial_project.retrieval.keywords.load import load_all_patient_keywords

eligible_trials_file = results_dir / "eligible_trials.parquet"


def get_trials_from_keywords(keywords) -> list[str]:
  if keywords is None:
    return []

  if isinstance(keywords, str):
    conditions = [keywords]
  else:
    conditions = [str(k).strip() for k in keywords if str(k).strip()]

  if not conditions:
    return []

  return rank_trials_for_conditions(
    conditions=conditions,
    bm25_wt=1,
    medcpt_wt=1,
    n_results=200,
  )

def save_trials_for_patient(patient_id, trial_ids):
  # save the trial ids for the patient for later retrieval
  try:
    df = pd.read_parquet(eligible_trials_file)
  except FileNotFoundError:
    df = pd.DataFrame(columns=["patient_id", "trial_ids"])

  if "patient_id" not in df.columns:
    df["patient_id"] = None
  if "trial_ids" not in df.columns:
    df["trial_ids"] = None

  mask = df["patient_id"] == patient_id
  trial_ids = list(trial_ids) if trial_ids is not None else []

  if mask.any():
    df.loc[mask, "trial_ids"] = [trial_ids]
  else:
    new_row = {"patient_id": patient_id, "trial_ids": trial_ids}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

  df.to_parquet(eligible_trials_file, index=False)

# idk if this works
def load_trials_for_patient(patient_id) -> list[str]:
  try:
    df = pd.read_parquet(eligible_trials_file)
  except FileNotFoundError:
    return []

  if "patient_id" not in df.columns or "trial_ids" not in df.columns:
    return []

  patient_rows = df[df["patient_id"] == patient_id]
  if patient_rows.empty:
    return []

  trial_ids = patient_rows.iloc[0]["trial_ids"]
  if trial_ids is None or pd.isna(trial_ids):
    return []

  if isinstance(trial_ids, list):
    return trial_ids

  if isinstance(trial_ids, tuple):
    return list(trial_ids)

  return [str(trial_ids)]

if __name__ == "__main__":
  patient_keywords = load_all_patient_keywords()
  for index, row in patient_keywords.iterrows():
    patient_id = row["patient_id"]
    keywords = row["keywords"]
    print(f"Patient {patient_id} has keywords: {keywords}")
    trial_ids = get_trials_from_keywords(keywords)
    print(f"Eligible trials for patient {patient_id}: {trial_ids}")
    save_trials_for_patient(patient_id, trial_ids)
  