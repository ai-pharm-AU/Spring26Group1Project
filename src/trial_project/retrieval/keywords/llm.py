"""
The llm for keyword generation
"""

from openai import OpenAI

def upload_context_files():
    # maybe move to or at least run in a setup script
  pass

def get_keywords(client: OpenAI, patient_info):
  return ["diabetes", "hypertension"]
  prompt = ""
  context_file_url=""
  model = "gpt-5.2"

  resp = client.responses.create(
    model=model,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                # TODO is this input_image?
                {"type": "input_image", "image_url": "context_file_url"},
                {"type": "input_text", "text": patient_info},
            ],
        }
    ],
  )

  # resp should be a list of keywords
  # return as array
  return [keyword for keyword in resp.output_text.split(",") if keyword]
    