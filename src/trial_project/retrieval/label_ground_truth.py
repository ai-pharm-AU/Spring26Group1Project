"""Main entry point for ground truth labeling tool."""

import argparse
from pathlib import Path

from trial_project.data.patients.load_patient import get_patient_json, load_all_patients
from trial_project.data.trials.load import _load_trials_dict
from trial_project.retrieval.labeling_session import LabelingSession
from trial_project.retrieval.labeling_cli import LabelingCLI
from trial_project.context import data_dir


def load_patient_ids_from_file(path: Path) -> list[str]:
    """Load patient IDs from a CSV or newline-separated file.
    
    Expects one patient ID per line or comma-separated on first line.
    """
    with open(path) as f:
        content = f.read().strip()
    
    # Try comma-separated first
    if "," in content:
        return [pid.strip() for pid in content.split(",")]
    
    # Otherwise, treat as newline-separated
    return [pid.strip() for pid in content.split("\n") if pid.strip()]


def load_all_patient_ids() -> list[str]:
    """Load all patient IDs from processed patients parquet."""
    patients_df = load_all_patients()
    if "Id" not in patients_df.columns:
        return []
    return patients_df["Id"].astype(str).tolist()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive ground truth labeling tool for patient-trial matches."
    )
    parser.add_argument(
        "--patient-ids",
        type=Path,
        help="Path to file with patient IDs (one per line or comma-separated). "
             "If not set, all patients with matches are used.",
    )
    parser.add_argument(
        "--trial-scope",
        choices=["matched", "all"],
        default="matched",
        help="'matched': label only retrieved trials from BM25 (default). "
             "'all': label patient against all trials (slower).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=data_dir / "processed_data" / "ground_truth_labels.parquet",
        help="Path to parquet file to resume from. "
             "Default: data/processed_data/ground_truth_labels.parquet",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Cap on number of pairs to label in this session (for testing).",
    )
    parser.add_argument(
        "--labeler-id",
        type=str,
        default="human",
        help="Identifier for the person labeling (default: 'human').",
    )
    
    args = parser.parse_args()
    
    # Load patient IDs if specified
    patient_ids = None
    if args.patient_ids:
        if not args.patient_ids.exists():
            print(f"✗ Patient IDs file not found: {args.patient_ids}")
            return
        patient_ids = load_patient_ids_from_file(args.patient_ids)
        print(f"Loaded {len(patient_ids)} patient IDs from {args.patient_ids}")
    
    # Load loaders
    def load_patient_wrapper(pid: str) -> dict:
        """Wrapper to load a single patient."""
        try:
            return get_patient_json(pid)
        except Exception:
            return {}
    
    def load_trials_wrapper() -> dict:
        """Wrapper to load all trials."""
        return _load_trials_dict()
    
    # Create session and CLI
    session = LabelingSession(
        patient_ids=patient_ids,
        trial_scope=args.trial_scope,
        resume_from=args.resume_from,
        load_patient_fn=load_patient_wrapper,
        load_trials_fn=load_trials_wrapper,
        load_all_patient_ids_fn=load_all_patient_ids,
        limit=args.limit,
    )
    
    cli = LabelingCLI(session, labeler_id=args.labeler_id)
    cli.run()


if __name__ == "__main__":
    main()
