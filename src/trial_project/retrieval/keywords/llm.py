"""
The llm for keyword generation
"""

from openai import OpenAI

prompt ="""
You are generating keywords based on patient records. 

Goal:
Given the patient evidence output from the evidence verification agent, generate a compact, comma-separated list of up to 32 key conditions that best represent the patient to be used for matching to relevant clinical trials. 

Important rules:
- Use only the verified patient evidence output that is provided. 
- Prefer claims that are supported or partially_supported and fit for eligibility use. 
- Do not use claims that are contradicted, not_found, or marked should_be_used_for_eligibility = false. 
- Prioritize clinically meaningful patient concepts that help retrieve or match relevant trials 
- Prefer persistent diseases, major chronic conditions, major recent acute conditions, major therapies, important procedures, meaningful biomarkers/labs, and high-signal demographic constraints when clearly relevant 
- Avoid low-value administrative notes unless they are directly useful for eligibility matching 
- Avoid generic words such as "patient", "history", "normal", "record", "follow-up", etc 
- Avoid duplicate terms, near-duplicates, and overly long phrases 
- Normalize obvious wording when helpful, but stay true to the verified evidence 
- If a concept is only a weak proxy, include it only if it is still likely to help matching and no stronger direct term is available 
- If there are more than 32 candidate terms, keep only the most discriminative and trial-relevant ones 

Available data sources: 

Verified patient evidence 

Prioritize:
1. Confirmed diseases and conditions 
2. Disease subtype, severity, stage, or chronicity if verified 
3. Current or important prior medications or treatment classes if likely relevant to eligibility 
4. Important procedures or surgeries 
5. High-signal lab or physiologic concepts if verified, such as elevated creatinine, hypertension, BMI/obesity, abnormal pulmonary function, smoking status, etc 
6. Core demographic constraints only when useful for matching, such as age group, sex, pregnancy-related status, or pediatric/adult status 
7. Social or logistical concepts only when they clearly affect trial eligibility or participation 

Avoid unless clearly relevant:
- Isolated wellness encounters
- Generic screenings
- Repeated measurements without a meaningful summary concept
- Lone units or dates in the final keyword list
- Broad filler terms that would match too many trials 

Output: 
- Output a single comma-separated line only.
- No numbering
- No bullets
- No JSON formatting
- No explanations 
- Each item should be either a single keyword or concise phrase, usually 1 to 4 words 
- Prefer normalized medical terms over raw EHR phrasing when the meaning is clear
- Use at most 32 items 

Selection guidance: 
- Include both disease names and especially useful companion concepts alongside one another when they add retrieval value, for example: 
  - "COPD, ex-smoker, fluticasone-salmeterol, albuterol" 
  - "type 2 diabetes, chronic kidney disease, creatinine, metformin" 
- If the patient evidence is sparse, return fewer keywords rather than weak filler terms 
- If the evidence contains both a broad and a specific term, prefer the more specific term unless both improve matching. 

Input:
"""

def get_keywords(client: OpenAI, patient_info):
  # print(f"Getting keywords for patient info: {patient_info}")
  prompt = 'You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. Please first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. The key condition list should be ranked by priority. Please output only a comma-separated list of the key conditions without any additional text or formatting.'
  model = "gpt-5-mini"
  resp = client.responses.create(
    model=model,
    instructions=prompt,
    input=patient_info
  )

  # resp should be a list of keywords
  # return as array
  return [keyword for keyword in resp.output_text.split(",") if keyword]
    