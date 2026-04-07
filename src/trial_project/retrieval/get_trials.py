"""
get the relevant trials for a patient based on keywords
"""

import pandas as pd
from trial_project.context import results_dir
from trial_project.retrieval.hybrid_fusion import rank_trials_for_conditions
from trial_project.retrieval.keywords.load import load_all_patient_keywords

eligible_trials_file = results_dir / "eligible_trials.parquet"


def _load_eligible_trials_df():
  if not eligible_trials_file.exists():
    return pd.DataFrame(columns=["patient_id", "trial_ids"])

  try:
    return pd.read_parquet(eligible_trials_file)
  except (FileNotFoundError, OSError, ValueError):
    return pd.DataFrame(columns=["patient_id", "trial_ids"])


def _normalize_trial_ids(trial_ids):
  if trial_ids is None:
    return []
  else:
    return trial_ids
  # ok maybe not the best but idek anymore
  if isinstance(trial_ids, (list, tuple, set)):
    return [str(trial_id).strip() for trial_id in trial_ids if str(trial_id).strip()]
  # if pd.isna(trial_ids):
  #   return []
  if isinstance(trial_ids, str):
    return [trial_ids.strip()] if trial_ids.strip() else []
  return [str(trial_ids).strip()]


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
  results_dir.mkdir(parents=True, exist_ok=True)
  df = _load_eligible_trials_df()

  if "patient_id" not in df.columns:
    df["patient_id"] = None
  if "trial_ids" not in df.columns:
    df["trial_ids"] = None

  normalized_trial_ids = _normalize_trial_ids(trial_ids)
  patient_id = str(patient_id)

  df = df[df["patient_id"].astype(str) != patient_id].copy()
  new_row = pd.DataFrame([
    {"patient_id": patient_id, "trial_ids": normalized_trial_ids}
  ])
  df = pd.concat([df, new_row], ignore_index=True)

  df.to_parquet(eligible_trials_file, index=False)

def load_trials_for_patient(patient_id) -> list[str]:
  df = _load_eligible_trials_df()

  if df.empty or "patient_id" not in df.columns or "trial_ids" not in df.columns:
    return []

  patient_rows = df[df["patient_id"].astype(str) == str(patient_id)]
  if patient_rows.empty:
    return []

  trial_ids = patient_rows.iloc[0]["trial_ids"]
  normalized_trial_ids = _normalize_trial_ids(trial_ids)
  print(f"Loaded trial ids for patient {patient_id}: {normalized_trial_ids}")
  return normalized_trial_ids

if __name__ == "__main__":
  patient_keywords = load_all_patient_keywords()
  for index, row in patient_keywords.iterrows():
    patient_id = row["patient_id"]
    keywords = row["keywords"]
    print(f"Patient {patient_id} has keywords: {keywords}")
    trial_ids = get_trials_from_keywords(keywords)
    print(f"Eligible trials for patient {patient_id}: {trial_ids}")
    save_trials_for_patient(patient_id, trial_ids)
  