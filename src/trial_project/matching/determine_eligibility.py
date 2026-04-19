"""
given a patient and a trial, determine criterion-level match results
"""

import argparse
import json
import logging

from trial_project.data.patients.load_patient import load_all_patients
from trial_project.data.trials.eligibility_verification import (
    load_trial_eligibility_verification,
)
from trial_project.matching.save_eligibility import (
    EligibilityDecision,
    load_saved_criterion_matches,
    save_criterion_matches,
    save_eligibility_decision,
)
from trial_project.matching.llm import (
    OverallTrialEligibilityLLMResult,
    TrialEligibilityLLMResult,
    evaluate_overall_trial_eligibility_llm,
    evaluate_trial_criteria_llm,
)
# from trial_project.matching.rule_based import is_patient_excluded_rule_based
from trial_project.retrieval.keywords.load import load_all_patient_keywords
from trial_project.retrieval.get_trials import load_trials_for_patient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _expected_criterion_ids(trial_eligibility_json: str | None) -> set[str]:
    if not trial_eligibility_json:
        return set()

    trial_payload = json.loads(trial_eligibility_json)
    atomic_criteria = trial_payload.get("atomic_criteria", [])
    return {
        str(item.get("criterion_id", "")).strip()
        for item in atomic_criteria
        if str(item.get("criterion_id", "")).strip()
    }


def _load_cached_criterion_result(
    patient_id: str,
    trial_id: str,
    criterion_model_name: str,
) -> TrialEligibilityLLMResult | None:
    trial_eligibility_json = load_trial_eligibility_verification(trial_id)
    expected_criterion_ids = _expected_criterion_ids(trial_eligibility_json)
    if not expected_criterion_ids:
        return None

    saved_matches = load_saved_criterion_matches(
        patient_id=patient_id,
        trial_id=trial_id,
        model_name=criterion_model_name,
    )
    if not saved_matches:
        return None

    saved_criterion_ids = {
        criterion.criterion_id.strip()
        for criterion in saved_matches
        if criterion.criterion_id.strip()
    }
    if saved_criterion_ids != expected_criterion_ids:
        return None

    return TrialEligibilityLLMResult(
        trial_id=trial_id,
        patient_id=patient_id,
        criterion_matches=saved_matches,
    )

overall_eligibility_prompt = """
You are determining overall trial eligibility from matched criterion results. 

  

Goal: 

Given criterion-level match results, determine the overall trial screening decision. 

  

Decision rules: 

- Use only the criterion-level match results provided. 

- Preserve inclusion vs exclusion exactly. 

- Do not invent evidence. 

- Do not treat missing evidence as negative evidence. 

- Overall decision rules: 

  - eligible = all inclusion criteria are meets and all exclusion criteria are not_excluded 

  - ineligible = any inclusion criterion is does_not_meet, or any exclusion criterion is excluded 

  - indeterminate = otherwise, when one or more required criteria remain insufficient_evidence 

- For confidence score: 

  - Output a numeric overall_confidence_score from 0.0 to 1.0. 

  - overall_confidence_score is the model's confidence in the assigned overall decision given only the available criterion-level evidence. 

- Base overall_confidence_score on the confidence of the full criterion set and whether the overall decision depends on any low-confidence or unresolved criteria. 

- Use the full 0.0 to 1.0 range and return a decimal with two digits. 

- Keep rationale short and evidence based. 

  

Instructions: 

1. Read the criterion-level match results. 

2. Identify whether any inclusion criterion has status does_not_meet. 

3. Identify whether any exclusion criterion has status excluded. 

4. Identify whether any required criteria remain insufficient_evidence. 

5. Use this rubric for confidence score:  

- 0.90 to 1.00 = explicit direct evidence with strong alignment, no meaningful conflict  

- 0.75 to 0.89 = strong evidence with minor ambiguity, minor incompleteness  

- 0.50 to 0.74 = moderate evidence, some ambiguity  

- 0.25 to 0.49 = weak evidence, substantial ambiguity, conflicting evidence, weak proxy use 

- 0.00 to 0.24 = very weak support 

6. Assign overall_confidence_score from 0.0 to 1.0 based on the strength and completeness of the criterion-level results. 

7. Keep rationale short and evidence based 

8. Return JSON only. 

  

Return JSON in this exact shape: 

{ 

  "trial_id": "", 

  "patient_id": "", 

  "overall_decision": "eligible|ineligible|indeterminate", 

  "overall_confidence_score": 0.0, 

  "overall_rationale": "", 

  "hard_stops": [], 

  "manual_review_flags": [], 

  "matching_notes": [] 

} 

  

Input criterion matches: 

{{criterion_matches_json}} 
"""

def determine_eligibility(
    patient_id: str,
    trial_id: str,
    model_name: str = "gpt-5-mini",
    overall_model_name: str | None = None,
) -> tuple[TrialEligibilityLLMResult, OverallTrialEligibilityLLMResult]:
    # returns criterion-level and overall match output for one patient/trial pair
  # rule_based_result = is_patient_excluded_rule_based(patient_id, trial_id)
  # if rule_based_result.eligible == False:
  #   return rule_based_result
  # otherwise check with llm
    criterion_model_name = model_name
    overall_model_name = overall_model_name or model_name

    criterion_result = _load_cached_criterion_result(
        patient_id=patient_id,
        trial_id=trial_id,
        criterion_model_name=criterion_model_name,
    )
    if criterion_result is None:
        criterion_result = evaluate_trial_criteria_llm(
            patient_id,
            trial_id,
            model_name=criterion_model_name,
        )

    overall_result = evaluate_overall_trial_eligibility_llm(
        patient_id=patient_id,
        trial_id=trial_id,
        criterion_matches=criterion_result.criterion_matches,
        overall_prompt=overall_eligibility_prompt,
        model_name=overall_model_name,
    )
    return criterion_result, overall_result


def main() -> int:
    """
    Load all patients, retrieve eligible trials, determine eligibility, and save results.
    """
    parser = argparse.ArgumentParser(
        description="Determine patient-trial eligibility and persist results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m trial_project.matching.determine_eligibility\n"
            "  python -m trial_project.matching.determine_eligibility --criterion-model-name gpt-5-mini --overall-model-name gpt-5.4\n"
            "  python -m trial_project.matching.determine_eligibility --model-name gpt-5-mini"
        ),
    )
    conflict_group = parser.add_mutually_exclusive_group()
    conflict_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rows for the same patient_id, trial_id, and model_name",
    )
    conflict_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing rows for the same patient_id, trial_id, and model_name",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-5-mini",
        help="Default model name used when the specific model flags are not provided",
    )
    parser.add_argument(
        "--criterion-model-name",
        default=None,
        help="Model name to use for criterion matching; falls back to --model-name if omitted",
    )
    parser.add_argument(
        "--overall-model-name",
        default=None,
        help="Model name to use for overall matching; falls back to --model-name if omitted",
    )
    args = parser.parse_args()

    conflict_policy = "overwrite" if args.overwrite else "skip"
    criterion_model_name = args.criterion_model_name or args.model_name
    overall_model_name = args.overall_model_name or args.model_name

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
    written_criterion_rows = 0
    skipped_criterion_rows = 0
    written_decision_rows = 0
    skipped_decision_rows = 0
    failed_pairs = 0

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
                criterion_result, overall_result = determine_eligibility(
                    patient_id,
                    trial_id,
                    model_name=criterion_model_name,
                    overall_model_name=overall_model_name,
                )

                written_count, skipped_count = save_criterion_matches(
                    patient_id=patient_id,
                    trial_id=trial_id,
                    criterion_matches=criterion_result.criterion_matches,
                    model_name=criterion_model_name,
                    conflict_policy=conflict_policy,
                )

                decision_written = save_eligibility_decision(
                    EligibilityDecision(
                        patient_id=patient_id,
                        trial_id=trial_id,
                        overall_decision=overall_result.overall_decision,
                        overall_confidence_score=overall_result.overall_confidence_score,
                        overall_rationale=overall_result.overall_rationale,
                        hard_stops=overall_result.hard_stops,
                        manual_review_flags=overall_result.manual_review_flags,
                        matching_notes=overall_result.matching_notes,
                        model_name=overall_model_name,
                        criteria_model=criterion_model_name,
                    ),
                    conflict_policy=conflict_policy,
                )

                written_criterion_rows += written_count
                skipped_criterion_rows += skipped_count
                if decision_written:
                    written_decision_rows += 1
                else:
                    skipped_decision_rows += 1
                logger.debug(
                    (
                        "Saved outputs for patient %s, trial %s "
                        "(criteria written=%s, criteria skipped=%s, decision_written=%s)"
                    ),
                    patient_id,
                    trial_id,
                    written_count,
                    skipped_count,
                    decision_written,
                )

            except Exception as e:
                failed_pairs += 1
                logger.error(f"Error determining criteria for patient {patient_id} and trial {trial_id}: {e}")
                continue

    logger.info(
        (
            "Completed! Criterion rows written: %s, skipped existing: %s, "
            "decision rows written: %s, decision rows skipped existing: %s, failed pairs: %s"
        ),
        written_criterion_rows,
        skipped_criterion_rows,
        written_decision_rows,
        skipped_decision_rows,
        failed_pairs,
    )
    return 0 if failed_pairs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())