"""
llm agents for determining patient trial eligibility
"""

from trial_project.api import generate_client
from trial_project.data.patients.load_patient import get_patient_llm_json
from trial_project.data.trials.eligibility import get_trial_eligibility_llm
from trial_project.data.trials.load import load_trial_json_llm
# from agents import Agent, Runner

client = generate_client()

# give patient info as str to avoid having to load and stuff every time
def is_patient_eligible_llm(patient_info_str, trial_id):
  trial_json = load_trial_json_llm(trial_id)
  trial_eligibility = get_trial_eligibility_llm(trial_id)

  prompt = """
You are matching structured trial criteria to a structured patient evidence profile. 

Goal: 

Compare one normalized trial extraction output to one normalized patient extraction output and produce criterion-level match results plus an overall trial screening decision. 

Decision rules: 
- Evaluate each atomic criterion separately. 
- Preserve inclusion vs exclusion exactly. 
- Use only the structured trial output and structured patient output provided. 
- Do not invent evidence. 
- Do not treat missing evidence as negative evidence. 
- If a criterion cannot be resolved from the patient evidence profile, mark it as insufficient_evidence. 

- For inclusion criteria: 
  - meets = direct support 
  - does_not_meet = direct contradiction 
  - insufficient_evidence = unresolved 

- For exclusion criteria: 
  - excluded = direct support that exclusion is present 
  - not_excluded = direct support that exclusion is not present, or evidence against it 
  - insufficient_evidence = unresolved 

Overall decision rules: 
- eligible = all inclusion criteria are meets and all exclusion criteria are not_excluded 
- ineligible = any inclusion criterion is does_not_meet, or any exclusion criterion is excluded 
- indeterminate = otherwise, when one or more required criteria remain insufficient_evidence 

Matching instructions: 
1. For each trial atomic criterion, inspect only the patient evidence categories needed for that criterion. 
2. Match using normalized concepts, synonyms, dates, numeric thresholds, and temporal constraints. 
3. If age is required, compute it from the patient birthdate relative to the trial screening date supplied externally. 
4. If timing matters, compare the criterion window to the patient evidence dates explicitly. 
5. If a proxy exists but is not definitive, include it under possible_proxies and keep the criterion unresolved unless the proxy is clearly sufficient. 
6. Keep rationale short and evidence-based. 
7. Return JSON only. 

Return JSON in this exact shape: 

{
  "trial_id": "", 
  "patient_id": "", 
  "overall_decision": "eligible|ineligible|indeterminate", 
  "overall_rationale": "", 
  "criterion_matches": [ 
    {
      "criterion_id": "", 
      "criterion_type": "inclusion|exclusion", 
      "criterion_text": "", 
      "status": "meets|does_not_meet|insufficient_evidence|excluded|not_excluded", 
      "matched_patient_evidence": [ 
        {
          "source_index": "demographics|condition_index|medication_index|procedure_index|observation_index|encounter_index", 
          "normalized_name": "", 
          "original_text": "", 
          "date": "", 
          "value": "", 
          "units": "" 
        } 
      ], 

      "possible_proxies": [], 
      "missing_but_needed": [], 
      "reasoning": "", 
      "confidence": "high|medium|low" 
    } 
  ], 

  "hard_stops": [], 
  "manual_review_flags": [], 
  "matching_notes": [] 

}
 """

  input = f"Trial eligibility criteria: {trial_eligibility}\nPatient evidence profile: {patient_info_str}"

  client = generate_client()
  response = client.responses.create(
    model="gpt-5-mini",
    instructions=prompt,
    input=input
)
  return response.output_text
  