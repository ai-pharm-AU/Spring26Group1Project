import pandas as pd
import json
from pathlib import Path
from trial_project.context import project_root

"""
Combines the synthea data from the various csv files into a single json file with one entry per patient. Each entry has id and various kept fields
"""

# ---- CONFIG ----
csv_dir = project_root / "data" / "synthea_sample_data_csv_latest"
out_file = project_root / "data" / "synthea_processed" / "all_patients_filtered_raw.json"

# ---- WHICH COLUMNS TO KEEP ----
KEEP_FIELDS = {
    "patients": ["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"],
    "encounters": ["START", "STOP", "ENCOUNTERCLASS", "DESCRIPTION"],
    "conditions": ["START", "STOP", "DESCRIPTION"],
    "medications": ["START", "STOP", "DESCRIPTION"],
    "observations": ["DATE", "DESCRIPTION", "VALUE", "UNITS"],
    "procedures": ["DATE", "DESCRIPTION"],
    "allergies": ["START", "STOP", "DESCRIPTION"],
    "immunizations": ["DATE", "DESCRIPTION"],
    "careplans": ["START", "STOP", "DESCRIPTION"]
}

# ---- LOAD TABLES ----
patients = pd.read_csv(csv_dir / "patients.csv", dtype=str)
patients = patients.head(20) # select only first 20 patients 

tables = {
    name: pd.read_csv(csv_dir / f"{name}.csv", dtype=str)
    for name in KEEP_FIELDS.keys()
    if name != "patients"
}

# ---- PATIENT ID COLUMN ----
patient_id_col = "Id"   # In Synthea patients.csv this is always "Id"

# Precompute patient ID column for each table
table_patient_cols = {}
for name, df in tables.items():
    for c in df.columns:
        if c.upper() == "PATIENT":
            table_patient_cols[name] = c
            break

# ---- BUILD JSON ----
all_records = []

for i, (_, p) in enumerate(patients.iterrows()):
    pid = p[patient_id_col]

    # Keep only selected patient columns
    patient_data = p[KEEP_FIELDS["patients"]].dropna().to_dict()
    record = {"_id": f"synthea-{i}", "patient": patient_data}

    for name, df in tables.items():
        pid_col = table_patient_cols[name]
        rows = df[df[pid_col] == pid]

        # Keep only selected columns
        cols = [c for c in KEEP_FIELDS[name] if c in rows.columns]
        rows = rows[cols]

        record[name] = rows.dropna(how="all").to_dict(orient="records")

    all_records.append(record)

# ---- WRITE SINGLE JSON FILE ----
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    json.dump(all_records, f, indent=2)

print("Done.")