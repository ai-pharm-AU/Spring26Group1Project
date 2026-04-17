"""
Async batch processing for patient-trial eligibility determination.
Uses ThreadPoolExecutor for concurrent processing of multiple patients.
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from trial_project.matching.determine_eligibility import determine_eligibility
from trial_project.matching.save_eligibility import save_eligibility_decision, EligibilityDecision
from trial_project.data.patients.load_patient import load_all_patients
from trial_project.retrieval.keywords.load import load_all_patient_keywords
from trial_project.retrieval.get_trials import load_trials_for_patient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single patient-trial pair."""
    patient_id: str
    trial_id: str
    success: bool
    skipped_existing: bool = False
    error: Optional[str] = None
    decision: Optional[EligibilityDecision] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BatchSummary:
    """Summary statistics for a batch processing run."""
    total_pairs: int
    written: int
    skipped_existing: int
    succeeded: int
    failed: int
    errors: List[Tuple[str, str, str]] = field(default_factory=list)  # (patient_id, trial_id, error_msg)
    elapsed_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Return success rate as percentage."""
        if self.total_pairs == 0:
            return 0.0
        return (self.succeeded / self.total_pairs) * 100


def async_determine_eligibility(
    patient_id: str,
    trial_id: str,
    conflict_policy: str = "skip",
    model_name: str = "gpt-5-mini",
) -> ProcessingResult:
    """
    Wrapper around determine_eligibility that catches exceptions and returns a ProcessingResult.
    """
    try:
        result = determine_eligibility(patient_id, trial_id, model_name=model_name)
        
        # Parse the LLM response (same logic as sync version)
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

        # Create eligibility decision
        decision = EligibilityDecision(
            patient_id=patient_id,
            trial_id=trial_id,
            eligible=eligible,
            exclusion_rule_hit=False,
            llm_checked=True,
            decision_source=decision_source,
            reasoning=reason,
            confidence=confidence,
            model_name=model_name,
            evaluated_at=datetime.utcnow()
        )

        # Save the decision (uses file locking internally)
        was_saved = save_eligibility_decision(decision, conflict_policy=conflict_policy)

        return ProcessingResult(
            patient_id=patient_id,
            trial_id=trial_id,
            success=True,
            skipped_existing=not was_saved,
            decision=decision
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing patient {patient_id}, trial {trial_id}: {error_msg}")
        return ProcessingResult(
            patient_id=patient_id,
            trial_id=trial_id,
            success=False,
            error=error_msg
        )


def batch_process_eligibilities(
    patient_ids: List[str],
    trial_dict: Dict[str, List[str]],
    max_workers: int = 5,
    conflict_policy: str = "skip",
    model_name: str = "gpt-5-mini",
) -> Tuple[List[ProcessingResult], BatchSummary]:
    """
    Process multiple patient-trial pairs concurrently using ThreadPoolExecutor.
    
    Args:
        patient_ids: List of patient IDs to process
        trial_dict: Dictionary mapping patient_id -> list of trial_ids
        max_workers: Maximum number of concurrent worker threads (default: 5)
        
    Returns:
        Tuple of (list of ProcessingResult objects, BatchSummary)
    """
    results = []
    start_time = time.time()
    
    # Prepare all tasks
    tasks = []
    for patient_id in patient_ids:
        trial_ids = trial_dict.get(patient_id, [])
        for trial_id in trial_ids:
            tasks.append((patient_id, trial_id))
    
    total_tasks = len(tasks)
    logger.info(f"Starting batch processing: {total_tasks} patient-trial pairs with {max_workers} workers")
    
    # Submit all tasks to executor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map patient_id, trial_id to futures
        future_to_task = {
            executor.submit(
                async_determine_eligibility,
                patient_id,
                trial_id,
                conflict_policy,
                model_name,
            ): (patient_id, trial_id)
            for patient_id, trial_id in tasks
        }
        
        # Process completed futures as they finish
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            try:
                result = future.result()
                results.append(result)
                
                # Log progress every 10 tasks
                if completed % max(1, total_tasks // 10) == 0 or completed == total_tasks:
                    logger.info(f"Progress: {completed}/{total_tasks} ({100*completed//total_tasks}%)")
                    
            except Exception as e:
                patient_id, trial_id = future_to_task[future]
                logger.error(f"Future raised exception for {patient_id}, {trial_id}: {e}")
                results.append(ProcessingResult(
                    patient_id=patient_id,
                    trial_id=trial_id,
                    success=False,
                    error=str(e)
                ))
    
    # Compute summary
    elapsed = time.time() - start_time
    written = sum(1 for r in results if r.success and not r.skipped_existing)
    skipped_existing = sum(1 for r in results if r.success and r.skipped_existing)
    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    errors = [(r.patient_id, r.trial_id, r.error) for r in results if not r.success]
    
    summary = BatchSummary(
        total_pairs=len(results),
        written=written,
        skipped_existing=skipped_existing,
        succeeded=succeeded,
        failed=failed,
        errors=errors,
        elapsed_time=elapsed
    )
    
    return results, summary


def process_all_patients_async(
    max_workers: int = 5,
    conflict_policy: str = "skip",
    model_name: str = "gpt-5-mini",
) -> BatchSummary:
    """
    Main orchestrator: Load all patients, retrieve trials, and process concurrently.
    
    Args:
        max_workers: Maximum number of concurrent worker threads (default: 5)
        
    Returns:
        BatchSummary with processing statistics
    """
    # Load all patients
    logger.info("Loading all patients...")
    patients_df = load_all_patients()
    logger.info(f"Loaded {len(patients_df)} patients")

    # Load patient keywords
    logger.info("Loading patient keywords...")
    keywords_df = load_all_patient_keywords()
    if keywords_df is not None and len(keywords_df) > 0:
        logger.info(f"Loaded keywords for {len(keywords_df)} patients")
    else:
        logger.warning("No patient keywords found")
        keywords_df = {}

    # Create dict for fast lookup
    keywords_dict = {}
    if hasattr(keywords_df, 'to_dict'):
        for _, row in keywords_df.iterrows():
            keywords_dict[row["patient_id"]] = row["keywords"]
    else:
        keywords_dict = keywords_df

    # Build trial dictionary
    logger.info("Building trial dictionary...")
    trial_dict = {}
    patient_ids_with_trials = []
    total_patients = len(patients_df)

    for idx, patient_row in patients_df.iterrows():
        patient_id = patient_row.get("Id") or patient_row.get("id")
        if not patient_id:
            logger.debug(f"Row {idx} has no patient ID, skipping")
            continue

        # Check if keywords exist for this patient
        keywords = keywords_dict.get(patient_id)
        if keywords is None or len(keywords) == 0:
            logger.debug(f"No keywords found for patient {patient_id}, skipping")
            continue

        # Load trials for this patient
        try:
            trial_ids = load_trials_for_patient(patient_id)
            if trial_ids and len(trial_ids) > 0:
                trial_dict[patient_id] = trial_ids
                patient_ids_with_trials.append(patient_id)
                if (idx + 1) % max(1, total_patients // 10) == 0:
                    logger.info(f"Trial retrieval progress: {idx + 1}/{total_patients}")
        except Exception as e:
            logger.error(f"Error retrieving trials for patient {patient_id}: {e}")
            continue

    logger.info(f"Prepared {len(patient_ids_with_trials)} patients with {sum(len(t) for t in trial_dict.values())} total trials")

    # Process all patient-trial pairs concurrently
    if patient_ids_with_trials:
        _, summary = batch_process_eligibilities(
            patient_ids_with_trials,
            trial_dict,
            max_workers=max_workers,
            conflict_policy=conflict_policy,
            model_name=model_name,
        )
        return summary
    else:
        logger.warning("No patients with trials found for processing")
        return BatchSummary(
            total_pairs=0,
            written=0,
            skipped_existing=0,
            succeeded=0,
            failed=0,
            elapsed_time=0.0,
        )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Async batch process patient-trial eligibility determination"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        choices=range(1, 21),
        help="Maximum number of concurrent workers (default: 5, range: 1-20)"
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

    logger.info(
        "Starting async eligibility determination with max_workers=%s, conflict_policy=%s, model_name=%s",
        args.max_workers,
        conflict_policy,
        args.model_name,
    )
    summary = process_all_patients_async(
        max_workers=args.max_workers,
        conflict_policy=conflict_policy,
        model_name=args.model_name,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("ASYNC PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total pairs processed: {summary.total_pairs}")
    logger.info(f"Written: {summary.written}")
    logger.info(f"Skipped existing: {summary.skipped_existing}")
    logger.info(f"Succeeded: {summary.succeeded}")
    logger.info(f"Failed: {summary.failed}")
    logger.info(f"Success rate: {summary.success_rate:.1f}%")
    logger.info(f"Elapsed time: {summary.elapsed_time:.2f}s")
    
    if summary.elapsed_time > 0:
        pairs_per_sec = summary.total_pairs / summary.elapsed_time
        logger.info(f"Throughput: {pairs_per_sec:.2f} pairs/sec")

    if summary.errors:
        logger.warning(f"\nFailed pairs ({len(summary.errors)}):")
        for patient_id, trial_id, error in summary.errors[:10]:  # Show first 10 errors
            logger.warning(f"  {patient_id} / {trial_id}: {error}")
        if len(summary.errors) > 10:
            logger.warning(f"  ... and {len(summary.errors) - 10} more errors")

    logger.info("=" * 60)
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    exit(main())
