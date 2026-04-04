"""
generate keywords for patient
should be cached etc
"""
import pandas as pd
from pandas import Series
from trial_project.data.patients.load_patient import get_patient_json, get_tables_dict
from trial_project.retrieval.keywords.llm import get_keywords
from trial_project.api import generate_client
from trial_project.context import data_dir

keywords_file = data_dir / "patient_keywords.parquet"

def generate_patient_keywords(patient_json):
  client = generate_client()
  keywords = get_keywords(client, patient_json)
  return keywords

# TODO maybe make keyed df or something
def save_patient_keywords(patient_id, keywords):
    df = load_all_patient_keywords()

    mask = df["patient_id"] == patient_id

    if mask.any():
        # Update existing
        df.loc[mask, "keywords"] = keywords
    else:
        # Insert new row
        new_row = {"patient_id": patient_id, "keywords": keywords}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_parquet(keywords_file, index=False)

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

def generate_patient_keywords_cached(patient_id: str):
  keywords_df = load_all_patient_keywords()
  if load_patient_keywords(patient_id, keywords_df) is not None:
    return load_patient_keywords(patient_id, keywords_df)
  else:
    patient = get_patient_json(patient_id, tables_dict=get_tables_dict())
    keywords = generate_patient_keywords(patient)
    save_patient_keywords(patient["id"], keywords)
    return keywords

if __name__ == "__main__":
  # load patient ids
  patients_ids = pd.read_parquet(data_dir / "processed_data/patients.parquet", columns=["Id"])
  tables_dict = get_tables_dict()
  for (index, patient_id) in patients_ids.iterrows():
    print(f"Generating keywords for patient {patient_id['Id']}")
    keywords = generate_patient_keywords_cached(patient_id['Id'])
    print(f"Keywords for patient {patient_id['Id']}: {keywords}")
  keywords = load_all_patient_keywords()
  print(keywords.head())
    # patient_json = get_patient_json(patient_id['Id'], tables_dict=tables_dict)
    # print(f"Patient JSON for patient {patient_id['Id']}: {patient_json}")
    # break
  
