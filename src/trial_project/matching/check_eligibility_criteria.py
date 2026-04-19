"""
given patient and trial, determine for each eligibility criteria whether the patient meets it, does not meet it, or if it's uncertain based on the evidence we have for the patient
"""

prompt = """
You are matching structured trial criteria to a structured patient evidence profile. 

  

Goal: 

Compare one normalized trial extraction output to one normalized patient extraction output and produce criterion-level match results. 

  

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

- For confidence score: 

  - For each criterion match, also output a numeric confidence_score from 0.0 to 1.0. 

  - confidence_score is the model's confidence in the assigned status given only the available structured evidence. 

- Base confidence_score on evidence quality, evidence directness, temporal/numeric alignment, internal consistency, and missingness according to the rules in the instructions. 

- Direct exact evidence should increase confidence_score. 

- Proxy evidence, conflicting evidence, ambiguous wording, and missing required evidence should lower confidence_score. 

- If the assigned status is insufficient_evidence, confidence_score should reflect confidence that insufficient_evidence is the correct status; it does not need to be low just because the evidence is missing. 

- Use the full 0.0 to 1.0 range and return a decimal with two digits. 

  

Matching instructions: 

1. For each trial atomic criterion, inspect only the patient evidence categories needed for that criterion. 

2. Match using normalized concepts, synonyms, dates, numeric thresholds, and temporal constraints 

3. If age is required, compute it from the patient birthdate relative to the trial screening date supplied externally. 

4. If timing is important, compare the criterion window to the patient evidence dates explicitly. 

5. If a proxy exists but is not definitive, include it under possible_proxies and keep the criterion unresolved unless the proxy is clearly sufficient. 

6. After assigning each criterion status, assign criterion_confidence_score from 0.0 to 1.0 based on how well the evidence supports that status. 

7. Use this rubric for confidence score: 

   - 0.90 to 1.00 = explicit direct evidence with strong alignment, no meaningful conflict 

   - 0.75 to 0.89 = strong evidence with minor ambiguity, minor incompleteness 

   - 0.50 to 0.74 = moderate evidence, some ambiguity 

   - 0.25 to 0.49 = weak evidence, substantial ambiguity, conflicting evidence, weak proxy use 

   - 0.00 to 0.24 = very weak support 

8. Keep rationale short and evidence based 

9. Return JSON only 

 

  

Return JSON in this exact shape: 

{ 

  "trial_id": "", 

  "patient_id": "", 

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

      "confidence": "high|medium|low", 

      "criterion_confidence_score": 0.0 

    } 

  ], 

  "manual_review_flags": [], 

  "matching_notes": [] 

} 

  

Input trial extraction: 

{{trial_extraction_json}} 

  

Input patient extraction: 

{{patient_extraction_json}} 

  

Optional screening reference date: 

{{screening_date}} 
"""