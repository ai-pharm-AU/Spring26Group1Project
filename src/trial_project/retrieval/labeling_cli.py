"""Interactive CLI interface for ground truth labeling."""

import sys
from datetime import datetime
from typing import Optional

from trial_project.retrieval.labeling_session import LabelingPair, LabelingSession
from trial_project.retrieval.labeling_storage import Label, save_labels


class LabelingCLI:
    """Interactive command-line interface for labeling patient-trial pairs."""

    def __init__(self, session: LabelingSession, labeler_id: str = "human"):
        """Initialize CLI with a labeling session."""
        self.session = session
        self.labeler_id = labeler_id
        self.pending_labels: list[Label] = []

    def run(self) -> None:
        """Main interaction loop."""
        self.session.initialize()
        
        if not self.session.pairs:
            print("No pairs to label. All done.")
            return
        
        print(f"\n{'='*80}")
        print(f"Ground Truth Labeling Tool")
        print(f"{'='*80}")
        print(f"Total pairs to label: {len(self.session.pairs)}")
        print(f"Already labeled: {len(self.session.labeled_pairs)}")
        print(f"\nCommands: [Y]es eligible  [N]o ineligible  [P]revious  [S]kip  [Q]uit")
        print(f"{'='*80}\n")
        
        try:
            while self.session.has_next():
                pair = self.session.current_pair()
                if pair is None:
                    break
                
                self._display_pair(pair)
                response = self._prompt_label()
                
                if response == "quit":
                    break
                elif response == "skip":
                    self.session.next_pair()
                elif response == "previous":
                    self.session.previous_pair()
                elif response in ("yes", "no"):
                    label = Label(
                        patient_id=pair.patient_id,
                        trial_id=pair.trial_id,
                        label=(response == "yes"),
                        labeled_at=datetime.now(),
                        labeler_id=self.labeler_id,
                    )
                    self.pending_labels.append(label)
                    self.session.next_pair()
            
            # Final save
            if self.pending_labels:
                self._handle_quit()
            else:
                print("No new labels to save.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving progress...")
            self._handle_quit()

    def _display_pair(self, pair: LabelingPair) -> None:
        """Display current pair for labeling."""
        print(f"\n{'-'*80}")
        print(f"Progress: {self.session.progress_string()}")
        print(f"{'-'*80}")
        
        # Patient section
        print("\nPATIENT:")
        self._display_patient(pair)
        
        # Trial section
        print("\nTRIAL:")
        self._display_trial(pair)
        
        print(f"\n{'-'*80}")

    def _display_patient(self, pair: LabelingPair) -> None:
        """Display patient summary."""
        patient = pair.patient_data
        
        # Extract basic demographics
        if "patient" in patient:
            demo = patient["patient"]
            age = self._calculate_age(demo.get("BIRTHDATE"))
            gender = demo.get("GENDER", "Unknown")
            race = demo.get("RACE", "Unknown")
            print(f"  ID: {pair.patient_id}")
            print(f"  Age: {age}, Gender: {gender}, Race: {race}")
        
        # Active conditions (last 3)
        conditions = patient.get("conditions", [])
        if conditions:
            print(f"  Conditions ({len(conditions)} total):")
            for cond in conditions[-3:]:
                desc = cond.get("DESCRIPTION", "Unknown")
                print(f"    - {desc}")
        
        # Current medications (last 3)
        meds = patient.get("medications", [])
        if meds:
            print(f"  Medications ({len(meds)} current):")
            for med in meds[-3:]:
                desc = med.get("DESCRIPTION", "Unknown")
                print(f"    - {desc}")
        
        # Recent observations (last 2)
        obs = patient.get("observations", [])
        if obs:
            print(f"  Recent Observations ({len(obs)} total):")
            for o in obs[-2:]:
                desc = o.get("DESCRIPTION", "Unknown")
                value = o.get("VALUE", "")
                units = o.get("UNITS", "")
                val_str = f"{value} {units}".strip()
                print(f"    - {desc}: {val_str}")

    def _display_trial(self, pair: LabelingPair) -> None:
        """Display trial summary."""
        trial = pair.trial_data
        
        print(f"  ID: {pair.trial_id}")
        
        # Title
        title = trial.get("Title") or trial.get("title") or "Unknown"
        print(f"  Title: {title}")
        
        # Description (truncate to 200 chars)
        desc = trial.get("Description") or trial.get("description") or ""
        if desc:
            if len(desc) > 200:
                desc = desc[:200] + "..."
            print(f"  Description: {desc}")
        
        # Eligibility criteria (first 300 chars)
        criteria = (
            trial.get("eligibility_criteria")
            or trial.get("EligibilityCriteria")
            or trial.get("eligibility")
            or ""
        )
        if criteria:
            if len(criteria) > 300:
                criteria = criteria[:300] + "..."
            print(f"  Eligibility Criteria:\n    {criteria}")

    def _prompt_label(self) -> Optional[str]:
        """Prompt user for label decision.
        
        Returns: "yes", "no", "skip", "previous", "quit", or None if invalid.
        """
        while True:
            try:
                response = input("\n➜ Label (y/n/p/s/q)? ").strip().lower()
                
                if response in ("y", "yes"):
                    return "yes"
                elif response in ("n", "no"):
                    return "no"
                elif response in ("p", "prev", "previous"):
                    if self.session.has_previous():
                        return "previous"
                    else:
                        print("  (Already at first pair)")
                        continue
                elif response in ("s", "skip"):
                    return "skip"
                elif response in ("q", "quit"):
                    return "quit"
                else:
                    print(f"  Invalid input '{response}'. Try: y/n/p/s/q")
            except EOFError:
                # Handle non-interactive mode (e.g., testing)
                return "quit"

    def _handle_quit(self) -> None:
        """Save pending labels and exit."""
        if not self.pending_labels:
            print("No new labels to save. Exiting.")
            return
        
        print(f"\nSaving {len(self.pending_labels)} labels...")
        try:
            save_labels(self.pending_labels, self.session.resume_from)
            print(f"Labels saved to: {self.session.resume_from}")
            print(f"Total labeled so far: {len(self.session.labeled_pairs) + len(self.pending_labels)}")
            self.pending_labels = []
        except Exception as e:
            print(f"Error saving labels: {e}", file=sys.stderr)
            raise

    def _calculate_age(self, birthdate: Optional[str]) -> str:
        """Calculate age from birthdate string (YYYY-MM-DD format)."""
        if not birthdate:
            return "Unknown"
        try:
            from datetime import datetime
            birth = datetime.strptime(birthdate, "%Y-%m-%d")
            today = datetime.today()
            age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            return str(age)
        except Exception:
            return "Unknown"
