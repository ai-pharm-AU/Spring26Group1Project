import pandas as pd
from trial_project.context import data_dir
from trial_project.data.patients.process import get_tables_dict
import json

patients_path = data_dir / "processed_data" / "patients.parquet"

def get_patient_llm_json(patient_id: str) -> str:
	# TODO actually process maybe
	patient_json = get_patient_json(patient_id, tables_dict=get_tables_dict())
	return json.dumps(patient_json)

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

	for table_name, table_df in tables_dict.items():
		if "PATIENT" not in table_df.columns:
			output[table_name] = []
			continue

		rows = table_df[table_df["PATIENT"] == patient_id]
		if rows.empty:
			output[table_name] = []
			continue

		table_cols = [c for c in rows.columns if c != "PATIENT"]
		output[table_name] = rows[table_cols].dropna(how="all").to_dict(orient="records")

	return output



 