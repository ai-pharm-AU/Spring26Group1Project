"""Parquet storage utilities for ground truth labeling."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class Label:
    """A human-provided label for a (patient, trial) pair."""

    patient_id: str
    trial_id: str
    label: bool  # True = eligible, False = ineligible
    labeled_at: datetime
    labeler_id: str = "human"

    def to_dict(self) -> dict:
        """Convert to dict for parquet serialization."""
        return {
            "patient_id": self.patient_id,
            "trial_id": self.trial_id,
            "label": self.label,
            "labeled_at": self.labeled_at,
            "labeler_id": self.labeler_id,
        }


def get_parquet_schema() -> pa.Schema:
    """Return PyArrow schema for labels parquet file."""
    return pa.schema(
        [
            pa.field("patient_id", pa.string()),
            pa.field("trial_id", pa.string()),
            pa.field("label", pa.bool_()),
            pa.field("labeled_at", pa.timestamp("us")),
            pa.field("labeler_id", pa.string()),
        ]
    )


def load_existing_labels(parquet_path: Path) -> pd.DataFrame:
    """Load existing labels from parquet file.
    
    Returns empty DataFrame if file doesn't exist.
    """
    if not parquet_path.exists():
        return pd.DataFrame(columns=["patient_id", "trial_id", "label", "labeled_at", "labeler_id"])
    
    return pd.read_parquet(parquet_path)


def get_labeled_pairs(parquet_path: Path) -> set[tuple[str, str]]:
    """Get set of (patient_id, trial_id) pairs already labeled.
    
    Returns empty set if file doesn't exist.
    """
    if not parquet_path.exists():
        return set()
    
    df = pd.read_parquet(parquet_path, columns=["patient_id", "trial_id"])
    return set(zip(df["patient_id"], df["trial_id"]))


def save_labels(labels: list[Label], parquet_path: Path) -> None:
    """Append labels to parquet file (creates if doesn't exist)."""
    if not labels:
        return
    
    # Convert labels to dict list
    data = [label.to_dict() for label in labels]
    df_new = pd.DataFrame(data)
    
    # Ensure timestamp is proper timezone-naive or -aware
    df_new["labeled_at"] = pd.to_datetime(df_new["labeled_at"])
    
    # Create parent directory if needed
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If file exists, append; otherwise create.
    if parquet_path.exists():
        df_existing = pd.read_parquet(parquet_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Keep one row per pair. If relabeled, latest write wins.
    df_combined = df_combined.drop_duplicates(subset=["patient_id", "trial_id"], keep="last")
    
    # Write with schema enforcement
    table = pa.Table.from_pandas(df_combined, schema=get_parquet_schema())
    pq.write_table(table, parquet_path)
