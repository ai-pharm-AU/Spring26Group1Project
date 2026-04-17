from trial_project.context import data_dir
import pandas as pd
from trial_project.data.patients.evidence.generate_evidence import get_patient_evidence

if __name__ == "__main__":
  patients_path = data_dir / "processed_data" / "patients.parquet"
  patients_df = pd.read_parquet(patients_path, columns=["Id"])

  for _, patient_row in patients_df.iterrows():
    patient_id = patient_row["Id"]
    print(f"Processing patient {patient_id}")
    patient_evidence = get_patient_evidence(patient_id)
    print(f"Patient evidence for {patient_id}: {patient_evidence}")
