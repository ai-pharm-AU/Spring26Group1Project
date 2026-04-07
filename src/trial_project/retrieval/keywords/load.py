import pandas as pd

from trial_project.context import data_dir

keywords_file = data_dir / "processed_data" / "patient_keywords.parquet"


def load_patient_keywords(patient_id, keywords_df=None):
    if keywords_df is None:
        try:
            keywords_df = pd.read_parquet(keywords_file)
        except FileNotFoundError:
            return None
    patient_rows = keywords_df[keywords_df["patient_id"] == patient_id]
    return patient_rows["keywords"].values[0] if len(patient_rows) > 0 else None


def load_all_patient_keywords():
    try:
        return pd.read_parquet(keywords_file)
    except FileNotFoundError:
        return pd.DataFrame(columns=["patient_id", "keywords"])
