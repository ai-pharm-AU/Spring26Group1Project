"""CLI for evaluation module."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from trial_project.evaluation.metrics import (
    compare_decisions_with_mismatches,
    print_metrics_report,
    print_mismatch_report,
    save_metrics,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM eligibility decisions against manual labels."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-5-mini",
        help="Overall model name to filter eligibility decisions (default: gpt-5-mini).",
    )
    parser.add_argument(
        "--criteria-model",
        type=str,
        default=None,
        help="Criteria model name to filter eligibility decisions (default: None, meaning no criteria model filtering).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for metrics JSON (default: results/).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output filename for metrics JSON (default: auto-generated with timestamp).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        logger.info(
            f"Starting evaluation with model_name='{args.model_name}'"
            + (f" and criteria_model='{args.criteria_model}'" if args.criteria_model else "")
        )

        # Compute metrics
        metrics, mismatches = compare_decisions_with_mismatches(
            model_name=args.model_name, criteria_model=args.criteria_model
        )

        # Determine output path
        output_path = None
        if args.output_file:
            output_path = args.output_file
        elif args.output_dir:
            # Auto-generate filename in output_dir
            from datetime import datetime

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = (
                args.output_dir / f"evaluation_metrics_{args.model_name}_{timestamp}.json"
            )

        # Save metrics
        saved_path = save_metrics(metrics, output_path)

        # Print report
        print_metrics_report(metrics)
        print_mismatch_report(mismatches)
        logger.info(f"Evaluation completed successfully. Results saved to {saved_path}")

    except ValueError as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
