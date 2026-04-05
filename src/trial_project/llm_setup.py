"""upload w/e files to the llm"""

from trial_project.api import generate_client
from trial_project.context import data_dir

client = generate_client()

if __name__ == "__main__":
  context_file_1_path = data_dir / "file.pdf"
  vector_store = client.vector_stores.create(name="trial data idk")
  client.vector_stores.upload_and_poll(vector_store.id, file=open(context_file_1_path, "rb"), metadata={"type": "trial_data"})
  print("Uploaded file to vector store")