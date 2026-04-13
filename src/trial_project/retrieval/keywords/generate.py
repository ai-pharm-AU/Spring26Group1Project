"""
generate keywords for patient
should be cached etc
"""
import pandas as pd
from pandas import Series
from trial_project.data.patients.load_patient import get_patient_llm_json, get_tables_dict
from trial_project.retrieval.keywords.llm import get_keywords
from trial_project.api import generate_client
from trial_project.context import data_dir
from trial_project.retrieval.keywords.load import load_all_patient_keywords, load_patient_keywords

keywords_file = data_dir / "processed_data" / "patient_keywords.parquet"

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

def generate_patient_keywords_cached(patient_id: str):
  keywords_df = load_all_patient_keywords()
  if load_patient_keywords(patient_id, keywords_df) is not None:
    return load_patient_keywords(patient_id, keywords_df)
  else:
    patient = get_patient_llm_json(patient_id)
    # print(patient)
    keywords = generate_patient_keywords(patient)
    save_patient_keywords(patient_id, keywords)
    return keywords

if __name__ == "__main__":
  # load patient ids
  patients_ids = pd.read_parquet(data_dir / "processed_data" / "patients.parquet", columns=["Id"])
  tables_dict = get_tables_dict()
  count = 0
  for (index, patient_id) in patients_ids.iterrows():
    count += 1
    print(f"Generating keywords for patient {patient_id['Id']}")
    keywords = generate_patient_keywords_cached(patient_id['Id'])
    print(f"Keywords for patient {patient_id['Id']}: {keywords}")
    print(f"Generated keywords for {count} patients so far")
  keywords = load_all_patient_keywords()
  print(keywords.head())
    # patient_json = get_patient_json(patient_id['Id'], tables_dict=tables_dict)
    # print(f"Patient JSON for patient {patient_id['Id']}: {patient_json}")
    # break
  
