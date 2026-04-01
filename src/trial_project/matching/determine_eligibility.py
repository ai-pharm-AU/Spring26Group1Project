"""
given a patient and a trial, determine if the patient is eligible for the trial
"""

from trial_project.matching.llm import is_patient_eligible_llm
from trial_project.matching.rule_based import is_patient_excluded_rule_based


def determine_eligibility(patient_id, trial_id):
  # returns True if patient is eligible for trial, False otherwise
  rule_based_result = is_patient_excluded_rule_based(patient_id, trial_id)
  if rule_based_result.eligible == False:
    return rule_based_result
  # otherwise check with llm
  return is_patient_eligible_llm(patient_id, trial_id)

