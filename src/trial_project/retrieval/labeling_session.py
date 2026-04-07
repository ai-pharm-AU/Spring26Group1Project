"""Session management for ground truth labeling."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from trial_project.context import results_dir, data_dir
from trial_project.retrieval.labeling_storage import get_labeled_pairs


@dataclass
class LabelingPair:
    """A (patient, trial) pair to label."""

    patient_id: str
    trial_id: str
    patient_data: dict  # Full patient JSON/dict
    trial_data: dict  # Full trial dict


class LabelingSession:
    """Manages the labeling session: filtering, tracking progress, and resumability."""

    def __init__(
        self,
        patient_ids: Optional[list[str]] = None,
        trial_scope: str = "matched",
        resume_from: Optional[Path] = None,
        load_patient_fn: Optional[Callable[[str], dict]] = None,
        load_trials_fn: Optional[Callable[[], dict]] = None,
        load_all_patient_ids_fn: Optional[Callable[[], list[str]]] = None,
        limit: Optional[int] = None,
    ):
        """Initialize labeling session.
        
        Args:
            patient_ids: List of patient IDs to label (None = all patients).
            trial_scope: "matched" (from eligible_trials.parquet) or "all" (all trials).
            resume_from: Path to parquet file to resume from. Defaults to standard location.
            load_patient_fn: Function to load single patient by ID.
            load_trials_fn: Function to load all trials dict.
            load_all_patient_ids_fn: Function to load all patient IDs.
            limit: Cap on number of pairs to offer in this session (for testing).
        """
        self.patient_ids = patient_ids
        self.trial_scope = trial_scope
        self.resume_from = resume_from or data_dir / "processed_data" / "ground_truth_labels.parquet"
        self.limit = limit
        
        # Placeholder for functions (will be replaced by actual loaders)
        self.load_patient_fn = load_patient_fn
        self.load_trials_fn = load_trials_fn
        self.load_all_patient_ids_fn = load_all_patient_ids_fn
        
        self.pairs: list[LabelingPair] = []
        self.current_index: int = 0
        self.labeled_pairs: set[tuple[str, str]] = set()
        
        self._initialized = False

    def initialize(self) -> None:
        """Load and filter pairs, track already-labeled ones."""
        if self._initialized:
            return
        
        # Load what's already been labeled
        self.labeled_pairs = get_labeled_pairs(self.resume_from)
        
        # Load trials once
        all_trials = self.load_trials_fn() if self.load_trials_fn else {}
        
        # Determine patient IDs to process
        if self.patient_ids:
            patient_ids_to_process = self.patient_ids
        else:
            patient_ids_to_process = self._get_all_patient_ids()
        
        # Build pairs based on trial_scope
        if self.trial_scope == "matched":
            pairs = self._build_matched_pairs(patient_ids_to_process, all_trials)
        elif self.trial_scope == "all":
            pairs = self._build_all_pairs(patient_ids_to_process, all_trials)
        else:
            raise ValueError(f"Unknown trial_scope: {self.trial_scope}")
        
        # Filter out already-labeled pairs
        unlabeled_pairs = [
            p for p in pairs
            if (p.patient_id, p.trial_id) not in self.labeled_pairs
        ]
        
        # Apply limit if specified
        if self.limit:
            unlabeled_pairs = unlabeled_pairs[:self.limit]
        
        self.pairs = unlabeled_pairs
        self._initialized = True

    def _get_all_patient_ids(self) -> list[str]:
        """Get all patient IDs based on selected trial scope."""
        if self.trial_scope == "all" and self.load_all_patient_ids_fn is not None:
            return self.load_all_patient_ids_fn()

        try:
            eligible_trials_path = results_dir / "eligible_trials.parquet"
            if eligible_trials_path.exists():
                df = pd.read_parquet(eligible_trials_path, columns=["patient_id"])
                return df["patient_id"].unique().tolist()
        except Exception:
            pass
        return []

    def _build_matched_pairs(self, patient_ids: list[str], all_trials: dict) -> list[LabelingPair]:
        """Build pairs from eligible_trials.parquet (retrieved matches)."""
        pairs = []
        eligible_trials_path = results_dir / "eligible_trials.parquet"
        
        if not eligible_trials_path.exists():
            return pairs
        
        df = pd.read_parquet(eligible_trials_path)
        df = df[df["patient_id"].isin(patient_ids)]
        
        for _, row in df.iterrows():
            patient_id = row["patient_id"]
            trial_ids = row.get("trial_ids", [])
            
            # trial_ids might be a list or a string representation.
            if isinstance(trial_ids, str):
                import ast
                import json

                try:
                    trial_ids = json.loads(trial_ids)
                except Exception:
                    trial_ids = ast.literal_eval(trial_ids)
            
            for trial_id in trial_ids:
                if trial_id in all_trials:
                    patient_data = self.load_patient_fn(patient_id) if self.load_patient_fn else {}
                    trial_data = all_trials.get(trial_id, {})
                    pairs.append(
                        LabelingPair(
                            patient_id=patient_id,
                            trial_id=trial_id,
                            patient_data=patient_data,
                            trial_data=trial_data,
                        )
                    )
        
        return pairs

    def _build_all_pairs(self, patient_ids: list[str], all_trials: dict) -> list[LabelingPair]:
        """Build all (patient, trial) combinations."""
        pairs = []
        
        for patient_id in patient_ids:
            patient_data = self.load_patient_fn(patient_id) if self.load_patient_fn else {}
            
            for trial_id, trial_data in all_trials.items():
                pairs.append(
                    LabelingPair(
                        patient_id=patient_id,
                        trial_id=trial_id,
                        patient_data=patient_data,
                        trial_data=trial_data,
                    )
                )
        
        return pairs

    def current_pair(self) -> Optional[LabelingPair]:
        """Get current pair to label, or None if done."""
        if self.current_index < len(self.pairs):
            return self.pairs[self.current_index]
        return None

    def has_next(self) -> bool:
        """Check if there are more pairs to label."""
        return self.current_index < len(self.pairs)

    def has_previous(self) -> bool:
        """Check if we can go back to a previous pair."""
        return self.current_index > 0

    def next_pair(self) -> Optional[LabelingPair]:
        """Move to next pair and return it."""
        if self.has_next():
            self.current_index += 1
            return self.current_pair()
        return None

    def previous_pair(self) -> Optional[LabelingPair]:
        """Move to previous pair and return it."""
        if self.has_previous():
            self.current_index -= 1
            return self.current_pair()
        return None

    def progress_string(self) -> str:
        """Return progress string like '5 / 100'."""
        total = len(self.pairs)
        current = self.current_index + 1 if self.current_pair() else self.current_index
        already_labeled = len(self.labeled_pairs)
        return f"{current} / {total} (already labeled: {already_labeled})"
