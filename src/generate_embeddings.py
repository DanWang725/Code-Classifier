import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from alive_progress import alive_bar
import sys
import torch
from utils.directories import prepared_dir
from utils.file_retrieval import DataFileDirectory
from utils.file_utils import prompt_save_file
torch.cuda.empty_cache()

checkpoint = "Salesforce/codet5p-110m-embedding"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

def loadCode(filename: str):
  source = pd.read_pickle(filename)
  return source

def get_embedding(text, tokenizer, model, chunk_size=512, overlap=50):
    # Tokenize the input text
    try:
      tokens = tokenizer(text, return_tensors='pt', truncation=False, padding=False).to(device)
      input_ids = tokens['input_ids'][0]
      
      # Initialize a list to store embeddings
      embeddings = []
      
      # Process the text in chunks
      for start_idx in range(0, len(input_ids), chunk_size - overlap):
          end_idx = start_idx + chunk_size
          chunk = input_ids[start_idx:end_idx]
          
          # If the chunk is smaller than chunk_size, pad it
          if len(chunk) < chunk_size:
              padding = torch.zeros(chunk_size - len(chunk), dtype=torch.long, device=device)
              chunk = torch.cat([chunk, padding])
          
          # Get the embedding for the chunk
          with torch.no_grad():
              # Pass the chunk through the model
              output = model(input_ids=chunk.unsqueeze(0))
              # The output is already the embedding tensor
              chunk_embedding = output[0]  # Use the first element of the output tuple
          
          # Store the embedding
          embeddings.append(chunk_embedding)
      
      # Aggregate the embeddings (e.g., by averaging)
      aggregated_embedding = torch.mean(torch.stack(embeddings), dim=0).cpu().numpy()
    except Exception as e:
      aggregated_embedding = None
       
    return aggregated_embedding

def generate(input_file: str):
  tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
  model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
  print("Loaded Model")

  source = loadCode(input_file)
  print(f"Loaded code. Count: {source.shape}")
  embeddings = source['code'].tolist()
  print(len(embeddings))
  output = source['actual label'].to_frame()

  with alive_bar(len(source)) as bar:
    for index, row in enumerate(embeddings):
      embeddings[index] = get_embedding(row, tokenizer, model, 1024)
      bar()

  output['code_embeddings'] = embeddings
  return output

if __name__ == "__main__":
  file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + prepared_dir
  question_files_class = DataFileDirectory(file_path, '.code.pkl')
  data_input_file = question_files_class.get_file("Select file to embed")

  output_map = prompt_save_file(question_files_class, '.emb.pkl')
  data_output_file = output_map[data_input_file]

  print("="*40)
  print("Reading from: ", data_input_file)
  print("Saving to: ", data_output_file)

  output = generate(data_input_file)
  output.to_pickle(data_output_file)
  print("Saved to ", data_output_file)
