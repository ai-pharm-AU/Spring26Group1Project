import pandas as pd
from trial_project.context import data_dir

keywords_file = data_dir / "patient_keywords.parquet" # not the best to have it twice, but need to avoid circular imports

def load_patient_keywords(patient_id, keywords_df=None):
  if keywords_df is None:
    try:
      keywords_df = pd.read_parquet(keywords_file)
    except FileNotFoundError:
      return None
  return keywords_df[keywords_df["patient_id"] == patient_id]["keywords"].values[0] if len(keywords_df[keywords_df["patient_id"] == patient_id]) > 0 else None

def load_all_patient_keywords():
  try:
    return pd.read_parquet(keywords_file)
  except FileNotFoundError:
    return pd.DataFrame(columns=["patient_id", "keywords"])