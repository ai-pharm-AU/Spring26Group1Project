"""CLI for exporting/importing manual labeling data."""

from __future__ import annotations

import argparse
from pathlib import Path

from trial_project.context import results_dir
from trial_project.labeling.storage import export_labeling_csv, import_labels_csv, manual_labels_file


def _build_export_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "export",
        help="Export matched patient-trial pairs to CSV for manual labeling.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=results_dir / "manual_labeling_export.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--matched-pairs-file",
        type=Path,
        default=None,
        help="Optional source file with pairs (csv/parquet). Defaults to retrieval output parquet.",
    )
    parser.add_argument(
        "--patient-ids",
        type=str,
        default=None,
        help="Optional comma-separated patient IDs filter.",
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=None,
        help="Optional subset pairs file with patient_id and trial_id columns.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Include already-labeled pairs in export.",
    )


def _build_import_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "import",
        help="Import labeled CSV rows into manual labels parquet.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input CSV containing labels.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=manual_labels_file,
        help="Output parquet path for manual labels.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["last", "first", "fail"],
        default="last",
        help="How to handle duplicate patient_id+trial_id rows in the import CSV.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build parser for labeling CLI."""
    parser = argparse.ArgumentParser(
        description="Export/import patient-trial labels for manual evaluation.",
        epilog="Note: To see specific options, run 'export --help' or 'import --help'.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_export_parser(subparsers)
    _build_import_parser(subparsers)
    return parser


def main() -> None:
    """Run labeling CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "export":
        output_path, count = export_labeling_csv(
            output_csv=args.output_csv,
            matched_pairs_source=args.matched_pairs_file,
            patient_ids=args.patient_ids,
            pairs_file=args.pairs_file,
            resume=not args.no_resume,
        )
        print(f"Exported {count} rows to {output_path}")
        return

    if args.command == "import":
        output_path, imported_count = import_labels_csv(
            input_csv=args.input_csv,
            output_path=args.output_parquet,
            duplicate_policy=args.duplicate_policy,
        )
        print(f"Imported {imported_count} labeled rows into {output_path}")
        return

    raise ValueError(f"Unknown command: {args.command}")
