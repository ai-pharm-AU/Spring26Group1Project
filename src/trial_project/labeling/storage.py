"""Persistence helpers for manual labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

import pandas as pd

from trial_project.context import results_dir
from trial_project.data.patients.load_patient import get_patient_json
from trial_project.data.trials.load import get_trial_json

label_file = results_dir / "manual_labels.parquet"
allowed_labels = {"eligible", "ineligible", "skip"}
manual_label_csv_columns = [
    "trial_id",
    "trial_json",
    "patient_id",
    "patient_json",
    "eligiblity",
    "notes",
]


@dataclass
class ManualLabel:
    patient_id: str
    trial_id: str
    label: str
    notes: str | None = None
    session_id: str | None = None
    labeled_at: datetime | None = None


def _empty_labels_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patient_id",
            "trial_id",
            "label",
            "notes",
            "session_id",
            "labeled_at",
        ]
    )


def load_labels(output_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(output_path) if output_path else label_file
    if not path.exists():
        return _empty_labels_df()

    try:
        df = pd.read_parquet(path)
    except Exception:
        return _empty_labels_df()

    for col in _empty_labels_df().columns:
        if col not in df.columns:
            df[col] = None

    return df[list(_empty_labels_df().columns)]


def labeled_pair_keys(output_path: str | Path | None = None) -> set[tuple[str, str]]:
    df = load_labels(output_path)
    if df.empty:
        return set()
    return set(
        zip(
            df["patient_id"].astype(str),
            df["trial_id"].astype(str),
        )
    )


def save_manual_label(label: ManualLabel, output_path: str | Path | None = None) -> None:
    normalized_label = label.label.strip().lower()
    if normalized_label not in allowed_labels:
        raise ValueError(f"Label must be one of {sorted(allowed_labels)}")

    path = Path(output_path) if output_path else label_file
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "patient_id": str(label.patient_id),
        "trial_id": str(label.trial_id),
        "label": normalized_label,
        "notes": label.notes,
        "session_id": label.session_id,
        "labeled_at": label.labeled_at or datetime.utcnow(),
    }

    df = load_labels(path)
    mask = (df["patient_id"].astype(str) == str(label.patient_id)) & (
        df["trial_id"].astype(str) == str(label.trial_id)
    )
    df = df.loc[~mask]

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_parquet(path, index=False)


def export_manual_labeling_csv(
    pairs_df: pd.DataFrame,
    output_csv_path: str | Path,
) -> dict[str, int]:
    """Export patient-trial pairs to a CSV for manual labeling."""
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pairs_df.empty:
        empty_df = pd.DataFrame(columns=manual_label_csv_columns)
        empty_df.to_csv(output_path, index=False)
        return {"queued": 0, "exported": 0, "failed": 0}

    rows: list[dict[str, object]] = []
    failed = 0

    for _, pair in pairs_df.iterrows():
        patient_id = str(pair["patient_id"]).strip()
        trial_id = str(pair["trial_id"]).strip()
        if not patient_id or not trial_id:
            failed += 1
            continue

        try:
            patient_json = get_patient_json(patient_id)
            trial_json = get_trial_json(trial_id)
        except Exception:
            failed += 1
            continue

        rows.append(
            {
                "trial_id": trial_id,
                "trial_json": json.dumps(trial_json, default=str),
                "patient_id": patient_id,
                "patient_json": json.dumps(patient_json, default=str),
                "eligiblity": "",
                "notes": "",
            }
        )

    out_df = pd.DataFrame(rows, columns=manual_label_csv_columns)
    out_df = out_df.drop_duplicates(subset=["patient_id", "trial_id"]).reset_index(drop=True)
    out_df.to_csv(output_path, index=False)

    return {"queued": len(pairs_df), "exported": len(out_df), "failed": failed}


def import_manual_labeling_csv(
    input_csv_path: str | Path,
    output_path: str | Path | None = None,
    conflict_policy: str = "skip",
) -> dict[str, int]:
    """Import manually labeled CSV rows into labels parquet."""
    csv_path = Path(input_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if conflict_policy not in {"skip", "overwrite"}:
        raise ValueError("conflict_policy must be one of: skip, overwrite")

    df = pd.read_csv(csv_path)
    eligibility_column = "eligiblity"
    if eligibility_column not in df.columns and "eligibility" in df.columns:
        eligibility_column = "eligibility"

    required_cols = {"trial_id", "trial_json", "patient_id", "patient_json", "notes", eligibility_column}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            "CSV must include columns: trial_id, trial_json, patient_id, patient_json, eligibility, notes"
        )

    total_rows = len(df)
    invalid_rows = 0
    unlabeled_rows = 0
    duplicate_rows = 0

    prepared_rows: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in df.iterrows():
        raw_patient_id = row["patient_id"]
        raw_trial_id = row["trial_id"]
        raw_label = row[eligibility_column]

        patient_id = "" if pd.isna(raw_patient_id) else str(raw_patient_id).strip()
        trial_id = "" if pd.isna(raw_trial_id) else str(raw_trial_id).strip()
        label = "" if pd.isna(raw_label) else str(raw_label).strip().lower()

        if not patient_id or not trial_id:
            invalid_rows += 1
            continue

        if not label:
            unlabeled_rows += 1
            continue

        if label not in allowed_labels:
            invalid_rows += 1
            continue

        raw_notes = row["notes"]
        notes: str | None
        if pd.isna(raw_notes):
            notes = None
        else:
            note_token = str(raw_notes).strip()
            notes = note_token if note_token else None

        key = (patient_id, trial_id)
        if key in prepared_rows:
            duplicate_rows += 1

        prepared_rows[key] = {
            "patient_id": patient_id,
            "trial_id": trial_id,
            "label": label,
            "notes": notes,
            "session_id": datetime.utcnow().strftime("manual-csv-%Y%m%d-%H%M%S"),
            "labeled_at": datetime.utcnow(),
        }

    if not prepared_rows:
        return {
            "total_rows": total_rows,
            "imported": 0,
            "skipped_existing": 0,
            "unlabeled": unlabeled_rows,
            "invalid": invalid_rows,
            "duplicate_rows": duplicate_rows,
        }

    parquet_path = Path(output_path) if output_path else label_file
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    existing_df = load_labels(parquet_path)
    existing_keys = set(zip(existing_df["patient_id"].astype(str), existing_df["trial_id"].astype(str)))

    rows_to_write = list(prepared_rows.values())
    skipped_existing = 0
    if conflict_policy == "skip":
        filtered_rows: list[dict[str, object]] = []
        for row in rows_to_write:
            key = (str(row["patient_id"]), str(row["trial_id"]))
            if key in existing_keys:
                skipped_existing += 1
                continue
            filtered_rows.append(row)
        rows_to_write = filtered_rows
    else:
        overwrite_keys = {(str(row["patient_id"]), str(row["trial_id"])) for row in rows_to_write}
        if overwrite_keys:
            existing_df = existing_df.loc[
                ~existing_df.apply(
                    lambda r: (str(r["patient_id"]), str(r["trial_id"])) in overwrite_keys,
                    axis=1,
                )
            ]

    if rows_to_write:
        existing_df = pd.concat([existing_df, pd.DataFrame(rows_to_write)], ignore_index=True)

    existing_df.to_parquet(parquet_path, index=False)

    return {
        "total_rows": total_rows,
        "imported": len(rows_to_write),
        "skipped_existing": skipped_existing,
        "unlabeled": unlabeled_rows,
        "invalid": invalid_rows,
        "duplicate_rows": duplicate_rows,
    }