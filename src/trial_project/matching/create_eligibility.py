"""
Interactive eligibility labeling tool for patient-trial pairs.

Go through all patients and trials and create eligibility labels for each pair manually.
Displays patient info, trial info, and prompts user to say if eligible or not.
Saves human labels to human_eligibility.parquet.
Does not overwrite existing labels automatically - shows them and allows confirmation or modification.
Can input array of patient ids and trial ids to label, then goes through each pair.
"""

# Yes AI did all this even more than usual

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from trial_project.context import results_dir
from trial_project.data.patients.load_patient import get_patient_json
from trial_project.data.trials.load import get_trial_json
from trial_project.matching.save_eligibility import EligibilityDecision


# ============================================================================
# PHASE 1: Core Data Loading & Display Helpers
# ============================================================================

def format_patient_for_display(patient_id: str) -> Optional[dict]:
    """
    Load patient data and return as dict for display.
    Returns None if patient not found (with warning printed).
    """
    try:
        patient_data = get_patient_json(patient_id)
        return patient_data
    except ValueError as e:
        print(f"⚠️  Warning: {e}")
        return None


def format_trial_for_display(trial_id: str) -> Optional[dict]:
    """
    Load trial data and return as dict for display.
    Returns None if trial not found (with warning printed).
    """
    try:
        trial_data = get_trial_json(trial_id)
        return trial_data
    except Exception as e:
        print(f"⚠️  Warning: Could not load trial {trial_id}: {e}")
        return None


def load_existing_label(patient_id: str, trial_id: str) -> Optional[EligibilityDecision]:
    """
    Check if a human label already exists for this pair in human_eligibility.parquet.
    Returns EligibilityDecision if found, None otherwise.
    """
    human_eligibility_file = results_dir / "human_eligibility.parquet"
    
    if not human_eligibility_file.exists():
        return None
    
    try:
        df = pd.read_parquet(human_eligibility_file)
        mask = (df["patient_id"] == patient_id) & (df["trial_id"] == trial_id)
        match = df[mask]
        
        if match.empty:
            return None
        
        row = match.iloc[0]
        decision = EligibilityDecision(
            patient_id=row["patient_id"],
            trial_id=row["trial_id"],
            eligible=row["eligible"],
            exclusion_rule_hit=row["exclusion_rule_hit"],
            llm_checked=row["llm_checked"],
            decision_source=row["decision_source"],
            reason=row["reason"],
            confidence=row["confidence"],
            model_name=row["model_name"],
            evaluated_at=row["evaluated_at"],
        )
        return decision
    except Exception as e:
        print(f"⚠️  Warning: Could not load existing label: {e}")
        return None


# ============================================================================
# PHASE 3: Persistence to human_eligibility.parquet
# ============================================================================

def save_human_eligibility_decision(decision: EligibilityDecision) -> None:
    """Upsert one human-labeled patient/trial row into human_eligibility.parquet."""
    human_eligibility_file = results_dir / "human_eligibility.parquet"
    
    row = {
        "patient_id": decision.patient_id,
        "trial_id": decision.trial_id,
        "eligible": decision.eligible,
        "exclusion_rule_hit": decision.exclusion_rule_hit,
        "llm_checked": decision.llm_checked,
        "decision_source": decision.decision_source,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "model_name": decision.model_name,
        "evaluated_at": decision.evaluated_at or datetime.utcnow(),
    }
    
    expected_columns = list(row.keys())
    
    if human_eligibility_file.exists():
        df = pd.read_parquet(human_eligibility_file)
    else:
        df = pd.DataFrame(columns=expected_columns)
    
    # Ensure all columns exist
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    
    # Remove existing row if present (upsert)
    mask = (df["patient_id"] == decision.patient_id) & (df["trial_id"] == decision.trial_id)
    df = df.loc[~mask]
    
    # Add new row
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_parquet(human_eligibility_file, index=False)


# ============================================================================
# PHASE 2: Interactive Labeling Loop
# ============================================================================

def label_eligibility_pairs(
    patient_ids: list[str],
    trial_ids: list[str],
    dry_run: bool = False
) -> dict:
    """
    Interactively label eligibility for all (patient_id, trial_id) pairs.
    
    Args:
        patient_ids: List of patient IDs to label
        trial_ids: List of trial IDs to label
        dry_run: If True, don't save to parquet, just show summary
    
    Returns:
        Summary dict with keys: labeled, skipped, total, already_existed
    """
    total_pairs = len(patient_ids) * len(trial_ids)
    labeled_count = 0
    skipped_count = 0
    already_existed_count = 0
    
    # Generate all pairs to label
    pairs_to_process = [
        (pid, tid) for pid in patient_ids for tid in trial_ids
    ]
    
    print(f"\n{'=' * 70}")
    print(f"Starting interactive eligibility labeling")
    print(f"Total pairs to process: {total_pairs}")
    print(f"{'=' * 70}\n")
    
    for idx, (patient_id, trial_id) in enumerate(pairs_to_process, 1):
        print(f"\n{'─' * 70}")
        print(f"Pair {idx}/{total_pairs}: Patient {patient_id} vs Trial {trial_id}")
        print(f"{'─' * 70}\n")
        
        # Load patient data
        patient_data = format_patient_for_display(patient_id)
        if patient_data is None:
            print("Skipping due to missing patient data.\n")
            skipped_count += 1
            continue
        
        # Load trial data
        trial_data = format_trial_for_display(trial_id)
        if trial_data is None:
            print("Skipping due to missing trial data.\n")
            skipped_count += 1
            continue
        
        # Display full data
        print("PATIENT DATA:")
        print(json.dumps(patient_data, indent=2, default=str))
        print("\nTRIAL DATA:")
        print(json.dumps(trial_data, indent=2, default=str))
        print()
        
        # Check for existing label
        existing_label = load_existing_label(patient_id, trial_id)
        
        if existing_label is not None:
            already_existed_count += 1
            print("EXISTING LABEL FOUND:")
            print(f"  Eligible: {existing_label.eligible}")
            print(f"  Source: {existing_label.decision_source}")
            print(f"  Reason: {existing_label.reason}")
            print(f"  Evaluated at: {existing_label.evaluated_at}")
            print()
            
            # Ask user if they want to keep it
            while True:
                response = input("Keep this label? (y/n/modify): ").strip().lower()
                if response in ["y", "n", "modify"]:
                    break
                print("Please enter 'y', 'n', or 'modify'")
            
            if response == "y":
                print(f"✓ Kept existing label for {patient_id} vs {trial_id}\n")
                continue
            elif response == "n":
                # User wants to overwrite
                # gee thanks AI
                pass
            else:  # modify
                # Show reason modification prompt below
                pass
        
        # Prompt for eligibility decision
        while True:
            eligible_input = input(
                "Is this patient eligible for the trial? (y/n/skip): "
            ).strip().lower()
            if eligible_input in ["y", "n", "skip"]:
                break
            print("Please enter 'y', 'n', or 'skip'")
        
        if eligible_input == "skip":
            print(f"⊘ Skipped {patient_id} vs {trial_id}\n")
            skipped_count += 1
            continue
        
        eligible = eligible_input == "y"
        
        # Prompt for optional reason
        reason = input("Reason (optional, press Enter to skip): ").strip()
        if not reason:
            reason = None
        
        # Create decision
        decision = EligibilityDecision(
            patient_id=patient_id,
            trial_id=trial_id,
            eligible=eligible,
            exclusion_rule_hit=False,  # User-labeled, not rule-based
            llm_checked=False,  # User-labeled, not LLM
            decision_source="manual_label",
            reason=reason,
            confidence=1.0,  # Human labels have confidence 1.0
            model_name=None,
            evaluated_at=datetime.utcnow(),
        )
        
        # Save or show that we would save
        if not dry_run:
            save_human_eligibility_decision(decision)
            status = "✓" if eligible else "✗"
            print(f"{status} Labeled {patient_id} vs {trial_id}: {eligible}")
            if reason:
                print(f"  Reason: {reason}")
            labeled_count += 1
        else:
            status = "✓" if eligible else "✗"
            print(f"[DRY RUN] Would save: {status} {patient_id} vs {trial_id}: {eligible}")
            if reason:
                print(f"  Reason: {reason}")
            labeled_count += 1
        
        print()
    
    # Print summary
    summary = {
        "labeled": labeled_count,
        "skipped": skipped_count,
        "total": total_pairs,
        "already_existed": already_existed_count,
    }
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total pairs processed: {summary['total']}")
    print(f"Newly labeled: {summary['labeled']}")
    print(f"Already existed: {summary['already_existed']}")
    print(f"Skipped: {summary['skipped']}")
    if not dry_run:
        print(f"Saved to: {results_dir / 'human_eligibility.parquet'}")
    else:
        print("DRY RUN - no data saved")
    print(f"{'=' * 70}\n")
    
    return summary


# ============================================================================
# PHASE 4: CLI Entry Point
# ============================================================================

def parse_id_list(id_string: str) -> list[str]:
    """
    Parse comma-separated or JSON array format of IDs.
    
    Examples:
        "P1,P2,P3" -> ["P1", "P2", "P3"]
        '["P1", "P2"]' -> ["P1", "P2"]
    """
    id_string = id_string.strip()
    
    # Try JSON array format first
    if id_string.startswith("[") and id_string.endswith("]"):
        try:
            return json.loads(id_string)
        except json.JSONDecodeError:
            pass
    
    # Try comma-separated format
    return [id.strip() for id in id_string.split(",") if id.strip()]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactively label patient-trial eligibility pairs"
    )
    parser.add_argument(
        "--patients",
        type=str,
        help="Comma-separated patient IDs or JSON array: 'P1,P2,P3' or '[\"P1\", \"P2\"]'",
    )
    parser.add_argument(
        "--trials",
        type=str,
        help="Comma-separated trial IDs or JSON array: 'T1,T2,T3' or '[\"T1\", \"T2\"]'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be saved without actually saving",
    )
    
    args = parser.parse_args()
    
    # If no args provided, ask user interactively
    if not args.patients or not args.trials:
        print("Interactive mode:")
        patient_input = input(
            "Enter patient IDs (comma-separated or JSON): "
        )
        trial_input = input(
            "Enter trial IDs (comma-separated or JSON): "
        )
    else:
        patient_input = args.patients
        trial_input = args.trials
    
    patient_ids = parse_id_list(patient_input)
    trial_ids = parse_id_list(trial_input)
    
    if not patient_ids:
        print("Error: No patient IDs provided")
        sys.exit(1)
    
    if not trial_ids:
        print("Error: No trial IDs provided")
        sys.exit(1)
    
    print(f"Patient IDs: {patient_ids}")
    print(f"Trial IDs: {trial_ids}")
    print()
    
    # Run labeling
    summary = label_eligibility_pairs(
        patient_ids,
        trial_ids,
        dry_run=args.dry_run,
    )