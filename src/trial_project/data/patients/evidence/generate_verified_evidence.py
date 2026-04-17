import argparse

import pandas as pd

from trial_project.context import data_dir
from trial_project.data.patients.evidence.generate_evidence import get_patient_evidence
from trial_project.data.patients.evidence.verify.llm import get_patient_evidence_verification


def load_all_patient_ids() -> list[str]:
	"""Load all patient IDs from the processed patient table."""
	patients_path = data_dir / "processed_data" / "patients.parquet"
	patients_df = pd.read_parquet(patients_path, columns=["Id"])
	return [str(pid) for pid in patients_df["Id"].dropna().astype(str).unique().tolist()]


def generate_and_verify_all_patients(
	model_name: str = "gpt-5-mini",
	use_cache: bool = True,
	continue_on_error: bool = False,
) -> tuple[int, int]:
	"""Generate and verify patient evidence for every patient ID."""
	patient_ids = load_all_patient_ids()
	total = len(patient_ids)

	succeeded = 0
	failed = 0

	for idx, patient_id in enumerate(patient_ids, start=1):
		print(f"[{idx}/{total}] Processing patient {patient_id}")
		try:
			get_patient_evidence(patient_id)
			get_patient_evidence_verification(
				patient_id=patient_id,
				model_name=model_name,
				use_cache=use_cache,
			)
			succeeded += 1
		except Exception as exc:
			failed += 1
			print(f"[{idx}/{total}] Failed patient {patient_id}: {exc}")
			if not continue_on_error:
				raise

	print(f"Completed patient evidence generation+verification: success={succeeded}, failed={failed}")
	return succeeded, failed


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate and verify patient evidence for all processed patients.",
	)
	parser.add_argument(
		"--model-name",
		default="gpt-5-mini",
		help="LLM model name used for verification.",
	)
	parser.add_argument(
		"--no-cache",
		action="store_true",
		help="Disable verification cache and re-run verification for each patient.",
	)
	parser.add_argument(
		"--continue-on-error",
		action="store_true",
		help="Continue processing remaining patients when one fails.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	_, failed = generate_and_verify_all_patients(
		model_name=args.model_name,
		use_cache=not args.no_cache,
		continue_on_error=args.continue_on_error,
	)
	return 0 if failed == 0 else 1


if __name__ == "__main__":
	raise SystemExit(main())
