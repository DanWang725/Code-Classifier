import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from alive_progress import alive_bar
import sys
import torch
torch.cuda.empty_cache()

data_input_file =  "../../data/prepared/testCode.pkl"

data_output_file = "../../data/prepared/embeddingsTest.pkl"


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

def main():
  tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
  model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
  print("Loaded Model")

  source = loadCode(data_input_file)
  print(f"Loaded code. Count: {source.shape}")
  embeddings = source['code'].tolist()
  output = source['actual label'].to_frame()

  with alive_bar(len(source)) as bar:
    
    for index, row in source.iterrows():
      embeddings[index] = get_embedding(row['code'], tokenizer, model, 1024)
      bar()

  output['code_embeddings'] = embeddings
  output.to_pickle(data_output_file)



# inputs = tokenizer.encode("def print_hello_world():\tprint('Hello World!')", return_tensors="pt").to(device)
# embedding = model(inputs)[0]

if __name__ == "__main__":
  print(torch.cuda.is_available())
  main()