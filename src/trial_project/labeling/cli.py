"""CLI for manual patient-trial labeling."""

from __future__ import annotations

import argparse
import sys
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
from trial_project.labeling.storage import (
    export_manual_labeling_csv,
    import_manual_labeling_csv,
    label_file,
)


def _add_pairs_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--matched-pairs-file",
        type=Path,
        default=matched_pairs_file,
        help="Matched pairs parquet with patient_id and trial_ids columns.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every matched pair after applying filters.",
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


def _resolve_filtered_pairs(args: argparse.Namespace) -> tuple[object, int]:
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
    return filtered_pairs, dropped_unmatched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual labeling utilities for patient-trial pairs."
    )
    subparsers = parser.add_subparsers(dest="command")

    review_parser = subparsers.add_parser(
        "review",
        help="Interactive manual labeling session.",
        description="Label matched patient-trial pairs interactively and save to parquet.",
    )
    _add_pairs_filters(review_parser)
    review_parser.add_argument(
        "--output",
        type=Path,
        default=label_file,
        help="Output parquet path for manual labels.",
    )
    review_parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume mode skips already-labeled pairs; disable to relabel pairs.",
    )

    export_parser = subparsers.add_parser(
        "csv-export",
        help="Export labeling CSV template.",
        description="Export patient-trial rows with trial/patient JSON for manual CSV labeling.",
    )
    _add_pairs_filters(export_parser)
    export_parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="CSV output path.",
    )

    import_parser = subparsers.add_parser(
        "csv-import",
        help="Import manually labeled CSV into parquet.",
        description="Import manual labels from CSV into the labels parquet.",
    )
    import_parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input CSV path to import.",
    )
    import_parser.add_argument(
        "--output",
        type=Path,
        default=label_file,
        help="Output parquet path for manual labels.",
    )
    import_parser.add_argument(
        "--conflict-policy",
        type=str,
        choices=["skip", "overwrite"],
        default="skip",
        help="Conflict handling for existing patient_id+trial_id rows.",
    )

    return parser


def main() -> None:
    # Backward compatibility: if no explicit subcommand is provided, default to review.
    if len(sys.argv) == 1:
        sys.argv.append("review")
    elif sys.argv[1] not in {"review", "csv-export", "csv-import", "-h", "--help"}:
        sys.argv.insert(1, "review")

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "csv-import":
        summary = import_manual_labeling_csv(
            input_csv_path=args.input_csv,
            output_path=args.output,
            conflict_policy=args.conflict_policy,
        )
        print("\nCSV import summary")
        print(f"Total rows: {summary['total_rows']}")
        print(f"Imported: {summary['imported']}")
        print(f"Skipped existing: {summary['skipped_existing']}")
        print(f"Skipped unlabeled: {summary['unlabeled']}")
        print(f"Skipped invalid: {summary['invalid']}")
        print(f"Deduped in CSV: {summary['duplicate_rows']}")
        print(f"Saved labels file: {args.output}")
        return

    filtered_pairs, dropped_unmatched = _resolve_filtered_pairs(args)

    if dropped_unmatched:
        print(f"Ignored {dropped_unmatched} requested pairs not found in matched pairs.")

    if args.command == "csv-export":
        summary = export_manual_labeling_csv(
            pairs_df=filtered_pairs,
            output_csv_path=args.output_csv,
        )
        print("\nCSV export summary")
        print(f"Queued pairs: {summary['queued']}")
        print(f"Exported rows: {summary['exported']}")
        print(f"Failed rows: {summary['failed']}")
        print(f"Saved csv file: {args.output_csv}")
        return

    if args.command is None:
        parser.error("Please provide one command: review, csv-export, or csv-import")

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