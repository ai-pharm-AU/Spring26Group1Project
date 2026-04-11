"""Interactive labeling loop for matched patient-trial pairs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trial_project.data.patients.load_patient import get_patient_json
from trial_project.data.trials.load import get_trial_json
from trial_project.labeling.storage import ManualLabel, labeled_pair_keys, save_manual_label


def _normalize_label(raw_label: str) -> str | None:
    token = raw_label.strip().lower()
    if token in {"eligible", "e"}:
        return "eligible"
    if token in {"ineligible", "i"}:
        return "ineligible"
    if token in {"skip", "s", "uncertain", "u"}:
        return "skip"
    return None


def _prompt_for_label() -> str | None:
    while True:
        raw = input("Label [e(ligible)/i(neligible)/s(kip)/q(uit)]: ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            confirm = input("Quit now? [y/N]: ").strip().lower()
            if confirm in {"y", "yes"}:
                return None
            continue

        normalized = _normalize_label(raw)
        if normalized is not None:
            return normalized

        print("Invalid label. Enter e for eligible, i for ineligible, s for skip, or q to quit.")


def run_review_session(
    pairs_df: pd.DataFrame,
    output_path: str | Path,
    resume: bool,
    session_id: str,
) -> dict[str, int]:
    if pairs_df.empty:
        print("No patient-trial pairs to review.")
        return {"queued": 0, "labeled": 0, "skipped_existing": 0}

    done_keys = labeled_pair_keys(output_path) if resume else set()
    pending_rows = []
    skipped_existing = 0

    for _, row in pairs_df.iterrows():
        key = (str(row["patient_id"]), str(row["trial_id"]))
        if key in done_keys:
            skipped_existing += 1
            continue
        pending_rows.append({"patient_id": key[0], "trial_id": key[1]})

    if not pending_rows:
        print("No unlabeled pairs remaining.")
        return {
            "queued": len(pairs_df),
            "labeled": 0,
            "skipped_existing": skipped_existing,
        }

    labeled_count = 0
    total_pending = len(pending_rows)

    for idx, pair in enumerate(pending_rows, start=1):
        patient_id = pair["patient_id"]
        trial_id = pair["trial_id"]

        print("\n" + "=" * 80)
        print(f"Pair {idx}/{total_pending}: patient_id={patient_id} trial_id={trial_id}")
        print("=" * 80)

        try:
            patient_json = get_patient_json(patient_id)
            trial_json = get_trial_json(trial_id)
        except Exception as exc:
            print(f"Failed to load pair data: {exc}")
            continue

        print("\nPATIENT JSON")
        print(json.dumps(patient_json, indent=2, sort_keys=True, default=str))

        print("\nTRIAL JSON")
        print(json.dumps(trial_json, indent=2, sort_keys=True, default=str))

        label = _prompt_for_label()
        if label is None:
            break

        notes_raw = input("Optional note (press Enter to skip): ").strip()
        notes = notes_raw if notes_raw else None

        save_manual_label(
            ManualLabel(
                patient_id=patient_id,
                trial_id=trial_id,
                label=label,
                notes=notes,
                session_id=session_id,
            ),
            output_path=output_path,
        )
        labeled_count += 1
        print(f"Saved label={label}. Progress: {labeled_count}/{total_pending}")

    return {
        "queued": len(pairs_df),
        "labeled": labeled_count,
        "skipped_existing": skipped_existing,
    }