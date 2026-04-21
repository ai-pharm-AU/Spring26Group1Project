"""CLI for patient trial ranking."""

from __future__ import annotations

import argparse
import logging

from trial_project.ranking.rank import rank_all_patients, rank_trials

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the ranking CLI."""
    parser = argparse.ArgumentParser(
        description="Rank matched trials for one patient or for all patients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Patient ID to rank trials for. If omitted, rank all patients.",
    )
    parser.add_argument(
        "--data-generation-model",
        default="gpt-5-mini",
        help="Model used to generate trial eligibility criteria.",
    )
    parser.add_argument(
        "--criteria-matching-model",
        default="gpt-5-mini",
        help="Model used to evaluate each criterion against patient evidence.",
    )
    parser.add_argument(
        "--overall-matching-model",
        default="gpt-5-mini",
        help="Model used to determine overall trial eligibility.",
    )
    conflict_group = parser.add_mutually_exclusive_group()
    conflict_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ranking rows for the same patient/trial/model combination.",
    )
    conflict_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trials that already have saved ranking rows for the same patient/trial/model combination.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main() -> int:
    """Rank matched trials for one patient or for all patients and print the ordered list."""
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    conflict_policy = "overwrite" if args.overwrite else "skip"

    if args.patient_id:
        ranked_by_patient = {
            args.patient_id: rank_trials(
                patient_id=args.patient_id,
                data_generation_model=args.data_generation_model,
                criteria_matching_model=args.criteria_matching_model,
                overall_matching_model=args.overall_matching_model,
                conflict_policy=conflict_policy,
            )
        }
    else:
        ranked_by_patient = rank_all_patients(
            data_generation_model=args.data_generation_model,
            criteria_matching_model=args.criteria_matching_model,
            overall_matching_model=args.overall_matching_model,
            conflict_policy=conflict_policy,
        )

    if not ranked_by_patient:
        logger.info("No ranked trials found")
        return 0

    for patient_id, ranked_trials in ranked_by_patient.items():
        if not ranked_trials:
            logger.info("No ranked trials found for patient %s", patient_id)
            continue

        print(f"patient_id={patient_id}")
        for index, ranking in enumerate(ranked_trials, start=1):
            print(
                f"  {index}. trial_id={ranking.trial_id} "
                f"overall_score={ranking.overall_score:.2f} "
                f"decision={ranking.overall_decision or 'unknown'} "
                f"relevance={ranking.condition_relevance_score:.2f} "
                f"benefit={ranking.potential_benefit_score:.2f} "
                f"safety={ranking.safety_score:.2f} "
                f"evidence={ranking.evidence_strength_score:.2f} "
                f"feasibility={ranking.feasibility_score:.2f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())