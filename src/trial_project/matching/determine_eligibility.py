"""
given a patient and a trial, determine if the patient is eligible for the trial
"""

import argparse
from trial_project.matching.llm import is_patient_eligible_llm
# from trial_project.matching.rule_based import is_patient_excluded_rule_based
import logging
from datetime import datetime

from trial_project.data.patients.load_patient import load_all_patients
from trial_project.retrieval.keywords.load import load_all_patient_keywords
from trial_project.retrieval.get_trials import load_trials_for_patient
from trial_project.matching.save_eligibility import save_eligibility_decision, EligibilityDecision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def determine_eligibility(patient_id, trial_id, model_name: str = "gpt-5-mini"):
  # returns True if patient is eligible for trial, False otherwise
  # rule_based_result = is_patient_excluded_rule_based(patient_id, trial_id)
  # if rule_based_result.eligible == False:
  #   return rule_based_result
  # otherwise check with llm
    return is_patient_eligible_llm(patient_id, trial_id, model_name=model_name)


def main() -> int:
    """
    Load all patients, retrieve eligible trials, determine eligibility, and save results.
    """
    parser = argparse.ArgumentParser(
        description="Determine patient-trial eligibility and persist results"
    )
    conflict_group = parser.add_mutually_exclusive_group()
    conflict_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rows with the same patient_id, trial_id, and model_name",
    )
    conflict_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing rows with the same patient_id, trial_id, and model_name (default)",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-5-mini",
        help="LLM model name to use for eligibility evaluation (default: gpt-5-mini)",
    )
    args = parser.parse_args()

    conflict_policy = "overwrite" if args.overwrite else "skip"

    # Load all patients
    logger.info("Loading all patients...")
    patients_df = load_all_patients()
    logger.info(f"Loaded {len(patients_df)} patients")

    # Load all patient trials
    logger.info("Loading patient keywords...")
    keywords_df = load_all_patient_keywords()
    if keywords_df is not None and len(keywords_df) > 0:
        logger.info(f"Loaded keywords for {len(keywords_df)} patients")
    else:
        logger.warning("No patient keywords found")
        keywords_df = {}

    # Create a dict for fast lookup
    keywords_dict = {}
    if hasattr(keywords_df, 'to_dict'):
        for _, row in keywords_df.iterrows():
            keywords_dict[row["patient_id"]] = row["keywords"]
    else:
        keywords_dict = keywords_df

    # Process each patient
    total_patients = len(patients_df)
    written_evaluations = 0
    skipped_evaluations = 0
    failed_evaluations = 0

    for idx, patient_row in patients_df.iterrows():
        patient_id = patient_row.get("Id") or patient_row.get("id")
        if not patient_id:
            logger.warning(f"Row {idx} has no patient ID, skipping")
            continue

        logger.info(f"[{idx + 1}/{total_patients}] Processing patient {patient_id}")

        # Get keywords for this patient TODO remove this part
        keywords = keywords_dict.get(patient_id)
        if keywords is None or len(keywords) == 0:
            logger.warning(f"No keywords found for patient {patient_id}, skipping")
            continue

        # Get eligible trials
        try:
            trial_ids = load_trials_for_patient(patient_id)
            logger.info(f"Found {len(trial_ids)} eligible trials for patient {patient_id}")
        except Exception as e:
            logger.error(f"Error retrieving trials for patient {patient_id}: {e}")
            continue

        # Determine eligibility for each trial
        for trial_id in trial_ids:
            try:
                result = determine_eligibility(
                    patient_id,
                    trial_id,
                    model_name=args.model_name,
                )
                
                # Parse the LLM response
                if isinstance(result, dict):
                    eligible = result.get("overall_decision") == "eligible"
                    reason = result.get("overall_rationale")
                    confidence = None  # Can extract from criterion_matches if needed
                    decision_source = "llm"
                else:
                    # Fallback if result is not a dict
                    eligible = bool(result)
                    reason = None
                    confidence = None
                    decision_source = "unknown"

                # Create and save eligibility decision
                decision = EligibilityDecision(
                    patient_id=patient_id,
                    trial_id=trial_id,
                    eligible=eligible,
                    exclusion_rule_hit=False,  # Can be updated if rule_based is used
                    llm_checked=True,
                    decision_source=decision_source,
                    reasoning=reason,
                    confidence=confidence,
                    model_name=args.model_name,
                    evaluated_at=datetime.utcnow()
                )

                was_saved = save_eligibility_decision(
                    decision,
                    conflict_policy=conflict_policy,
                )
                if was_saved:
                    written_evaluations += 1
                    logger.debug(f"Saved eligibility decision for patient {patient_id} and trial {trial_id}")
                else:
                    skipped_evaluations += 1
                    logger.debug(
                        f"Skipped existing eligibility decision for patient {patient_id} and trial {trial_id}"
                    )

            except Exception as e:
                failed_evaluations += 1
                logger.error(f"Error determining eligibility for patient {patient_id} and trial {trial_id}: {e}")
                continue

    logger.info(
        "Completed! Written: %s, Skipped existing: %s, Failed: %s",
        written_evaluations,
        skipped_evaluations,
        failed_evaluations,
    )
    return 0 if failed_evaluations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())