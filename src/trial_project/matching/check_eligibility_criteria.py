"""Prompt builder for criterion-level trial matching."""

from trial_project.data.patients.process import GENERATION_DATE


def get_criterion_matching_prompt() -> str:
    """Return instructions for criterion-only matching between trial and patient evidence."""
    screening_date = GENERATION_DATE.date().isoformat()
    return """
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
  - For each criterion match, output a numeric confidence score from 0.0 to 1.0.
  - Score confidence in the assigned status given only the available structured evidence.
  - Use the full 0.0 to 1.0 range and return a decimal with two digits when possible.

Matching instructions:
1. For each trial atomic criterion, inspect only the patient evidence categories needed for that criterion.
2. Match using normalized concepts, synonyms, dates, numeric thresholds, and temporal constraints.
3. If age is required, compute it from patient birthdate relative to the supplied screening date.
4. If timing is important, compare criterion windows to patient evidence dates explicitly.
5. If a proxy exists but is not definitive, include it under possible_proxies and keep the criterion unresolved unless clearly sufficient.
6. Keep rationale short and evidence based.
7. Return JSON only.

Output contract:
- Use the provided schema exactly.
- Populate all required fields.
- Do not add extra keys.

Input trial extraction:
{{trial_extraction_json}}

Input patient extraction:
{{patient_extraction_json}}

Optional screening reference date:
{{screening_date}}
""".replace("{{screening_date}}", screening_date)