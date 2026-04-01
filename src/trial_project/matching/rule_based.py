"""
Quick rule based exclusion for trial
"""

from trial_project.matching.save_eligibility import EligibilityDecision


def check_single_rule(patient_id, trial_id, field, condition, value) -> bool:
  # compare field value for patient and trial, return True if condition is satisfied
  return False

def is_patient_excluded_rule_based(patient_id, trial_id) -> EligibilityDecision:
  # check if patient excluded based on the rules
  # returns True if patient is excluded, False otherwise
  return EligibilityDecision(
    patient_id=patient_id,
    trial_id=trial_id,
    eligible=False,
    exclusion_rule_hit=True,
    llm_checked=False,
    decision_source="rule_based",
    reason="rule-matched"
  )