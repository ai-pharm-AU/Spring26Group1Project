"""CLI for manual patient-trial labeling."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from trial_project.labeling.pairs import (
    filter_pairs,
    load_matched_pairs,
    load_pairs_subset_file,
    matched_pairs_file,
    parse_patient_ids,
)
from trial_project.labeling.review import run_review_session
from trial_project.labeling.storage import label_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label matched patient-trial pairs and save manual labels to parquet."
    )
    parser.add_argument(
        "--matched-pairs-file",
        type=Path,
        default=matched_pairs_file,
        help="Matched pairs parquet with patient_id and trial_ids columns.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Label every matched pair after applying resume behavior.",
    )
    parser.add_argument(
        "--patient-ids",
        type=str,
        default="",
        help="Comma-separated patient IDs subset.",
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=None,
        help="Optional CSV or parquet with patient_id and trial_id columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=label_file,
        help="Output parquet path for manual labels.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume mode skips already-labeled pairs; disable to relabel pairs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    matched_pairs = load_matched_pairs(args.matched_pairs_file)
    patient_ids = parse_patient_ids(args.patient_ids)
    subset_pairs_df = load_pairs_subset_file(args.pairs_file) if args.pairs_file else None

    if not args.all and not patient_ids and subset_pairs_df is None:
        print("No subset flags provided; defaulting to all matched pairs.")

    filtered_pairs, dropped_unmatched = filter_pairs(
        matched_pairs=matched_pairs,
        patient_ids=patient_ids,
        subset_pairs_df=subset_pairs_df,
    )

    if dropped_unmatched:
        print(f"Ignored {dropped_unmatched} requested pairs not found in matched pairs.")

    session_id = datetime.utcnow().strftime("manual-%Y%m%d-%H%M%S")
    print(
        f"Starting labeling session {session_id} with {len(filtered_pairs)} queued pairs "
        f"(resume={args.resume})."
    )

    summary = run_review_session(
        pairs_df=filtered_pairs,
        output_path=args.output,
        resume=args.resume,
        session_id=session_id,
    )

    print("\nLabeling summary")
    print(f"Queued pairs: {summary['queued']}")
    print(f"Labeled this run: {summary['labeled']}")
    print(f"Skipped existing: {summary['skipped_existing']}")
    print(f"Saved labels file: {args.output}")


if __name__ == "__main__":
    main()