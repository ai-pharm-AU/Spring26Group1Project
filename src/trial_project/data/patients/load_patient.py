import json

import pandas as pd

from trial_project.context import data_dir
from trial_project.data.patients.process import get_tables_dict

patients_path = data_dir / "processed_data" / "patients.parquet"

ENCOUNTER_LINKED_TABLES = [
    "conditions",
    "medications",
    "observations",
    "procedures",
    "allergies",
    "immunizations",
    "careplans",
]


def load_all_patients() -> pd.DataFrame:
    return pd.read_parquet(patients_path)


# TODO this could be a field or saved somewhere
def get_patient_llm_json(patient_id: str) -> str:
    # TODO actually process maybe
    patient_json = get_patient_json(patient_id, tables_dict=get_tables_dict())
    return json.dumps(_replace_nan_values(patient_json))


def _replace_nan_values(value):
    if isinstance(value, dict):
        return {key: _replace_nan_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_nan_values(item) for item in value]
    if pd.isna(value):
        return ""
    return value


def get_patient_json(patient_id: str, tables_dict=None) -> dict:
    """Load one patient and all related records as a JSON-serializable dict."""
    patients_df = pd.read_parquet(patients_path)

    patient_rows = patients_df[patients_df["Id"] == patient_id]
    if patient_rows.empty:
        raise ValueError(f"Patient not found: {patient_id}")

    patient_data = patient_rows.iloc[0].dropna().to_dict()
    output = {"id": patient_id, "patient": patient_data}

    if tables_dict is None:
        tables_dict = get_tables_dict()

    encounters_df = tables_dict.get("encounters", pd.DataFrame())
    encounters = []
    encounter_map = {}

    if "PATIENT" in encounters_df.columns:
        encounter_rows = encounters_df[encounters_df["PATIENT"] == patient_id]
        for _, row in encounter_rows.iterrows():
            encounter_dict = row.dropna().to_dict()
            encounter_payload = {k: v for k, v in encounter_dict.items() if k != "PATIENT"}
            for table_name in ENCOUNTER_LINKED_TABLES:
                encounter_payload[table_name] = []

            encounter_id = encounter_payload.get("Id")
            if not pd.isna(encounter_id):
                encounter_map[str(encounter_id)] = encounter_payload

            encounters.append(encounter_payload)

    output["encounters"] = encounters

    for table_name, table_df in tables_dict.items():
        if table_name == "encounters":
            continue

        if "PATIENT" not in table_df.columns:
            output[table_name] = []
            continue

        rows = table_df[table_df["PATIENT"] == patient_id]
        if rows.empty:
            output[table_name] = []
            continue

        if table_name in ENCOUNTER_LINKED_TABLES and "ENCOUNTER" in rows.columns and encounter_map:
            encounter_keys = rows["ENCOUNTER"].astype(str)
            linked_mask = rows["ENCOUNTER"].notna() & encounter_keys.isin(encounter_map)

            linked_rows = rows[linked_mask]
            if not linked_rows.empty:
                for encounter_key, encounter_rows in linked_rows.groupby(linked_rows["ENCOUNTER"].astype(str)):
                    table_cols = [c for c in encounter_rows.columns if c not in ["PATIENT", "ENCOUNTER"]]
                    encounter_map[encounter_key][table_name] = (
                        encounter_rows[table_cols].dropna(how="all").to_dict(orient="records")
                    )

            unlinked_rows = rows[~linked_mask]
            if unlinked_rows.empty:
                output[table_name] = []
            else:
                table_cols = [c for c in unlinked_rows.columns if c != "PATIENT"]
                output[table_name] = (
                    unlinked_rows[table_cols].dropna(how="all").to_dict(orient="records")
                )
            continue

        table_cols = [c for c in rows.columns if c != "PATIENT"]
        output[table_name] = rows[table_cols].dropna(how="all").to_dict(orient="records")

    return output