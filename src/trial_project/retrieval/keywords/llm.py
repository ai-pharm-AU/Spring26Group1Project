"""
The llm for keyword generation
"""

from openai import OpenAI


def get_keywords(client: OpenAI, patient_info):
  print(f"Getting keywords for patient info: {patient_info}")
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
    