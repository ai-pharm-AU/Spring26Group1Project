import pandas as pd
from pathlib import Path
from trial_project.context import data_dir

# ---- CONFIG ----
csv_dir = data_dir / "synthea_sample_data_csv_latest"
out_dir = data_dir / "processed_data"
out_dir.mkdir(parents=True, exist_ok=True)

# ---- WHICH COLUMNS TO KEEP ----
KEEP_FIELDS = {
    "patients": ["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"],
    "encounters": ["PATIENT", "START", "STOP", "ENCOUNTERCLASS", "DESCRIPTION"],
    "conditions": ["PATIENT", "START", "STOP", "DESCRIPTION"],
    "medications": ["PATIENT", "START", "STOP", "DESCRIPTION"],
    "observations": ["PATIENT", "DATE", "DESCRIPTION", "VALUE", "UNITS"],
    "procedures": ["PATIENT", "DATE", "DESCRIPTION"],
    "allergies": ["PATIENT", "START", "STOP", "DESCRIPTION"],
    "immunizations": ["PATIENT", "DATE", "DESCRIPTION"],
    "careplans": ["PATIENT", "START", "STOP", "DESCRIPTION"]
}

def get_tables_dict():
    tables_dict = {}

    for name in KEEP_FIELDS:
        if name == "patients":
            continue

        file_path = out_dir / f"{name}.parquet"

        if file_path.exists():
            tables_dict[name] = pd.read_parquet(file_path)

    return tables_dict

def load_synthea_tables(n_patients=None):
    # ---- LOAD PATIENTS ----
    patients = pd.read_csv(csv_dir / "patients.csv", dtype=str)
    patients = patients[KEEP_FIELDS["patients"]]
    if n_patients is not None:
        patients = patients.head(n_patients)

    # Get selected patient IDs
    patient_ids = set(patients["Id"])

    # ---- LOAD AND FILTER OTHER TABLES ----
    tables = {}

    for name in KEEP_FIELDS:
        if name == "patients":
            continue

        df = pd.read_csv(csv_dir / f"{name}.csv", dtype=str)

        # Ensure PATIENT column exists
        if "PATIENT" not in df.columns:
            raise ValueError(f"{name} missing PATIENT column")

        # Filter to selected patients
        df = df[df["PATIENT"].isin(patient_ids)]

        # Keep only desired columns (if they exist)
        cols = [c for c in KEEP_FIELDS[name] if c in df.columns]
        df = df[cols]

        tables[name] = df

    return patients, tables

def save_tables(patients, tables):
    # Save patients
    patients.to_parquet(out_dir / "patients.parquet", index=False)

    # Save each table
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


if __name__ == "__main__":
    patients_df, tables_dict = load_synthea_tables(n_patients=20)
    save_tables(patients_df, tables_dict)
