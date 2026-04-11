"""Persistence helpers for manual labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from trial_project.context import results_dir

label_file = results_dir / "manual_labels.parquet"
allowed_labels = {"eligible", "ineligible", "skip"}


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