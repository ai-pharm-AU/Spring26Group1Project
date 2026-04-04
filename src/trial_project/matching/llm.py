"""
llm agents for determining patient trial eligibility
"""

from trial_project.api import generate_client
from trial_project.data.patients.load_patient import get_patient_json_llm
from trial_project.data.trials.load import load_trial_json_llm
from agents import Agent, Runner

client = generate_client()

# give patient info as str to avoid having to load and stuff every time
def is_patient_eligible_llm(patient_info_str, trial_id):
  trial_json = load_trial_json_llm(trial_id)

  # now give to llm along with output format and context

  main_agent = Agent()
  evaluator_agent = Agent()
  
  eligible = False;
  eligibility_score = 0.1;
  return [eligible, eligibility_score]
  pass