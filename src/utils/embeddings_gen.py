from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch
from utils.file_retrieval import DataFileDirectory
from utils.embedding import insert_df

device="cuda"

def get_embedding_roBERTa(text, tokenizer, model):
    try:
      inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
      with torch.no_grad():
        output = model(**inputs)
      embedding = output.last_hidden_state[:, 0, :].squeeze(0)
      return embedding.cpu().numpy()
    except Exception as e:
      print(e)
      return None

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

def embed_files(dir: DataFileDirectory, tokenizer, model) -> pd.DataFrame:
  output = pd.DataFrame(columns=['code_embeddings', 'actual label'])
  for path, name in dir.get_path_to_file_name_mapping().items():
    code = open(path, 'r', encoding='utf-8').read()
    embedding = get_embedding(code, tokenizer, model)
    # label = input(f'{name}: is this human (y/n)')
    # if(label == "y"):
    #    actual_label = 'human'
    # else:
    #    actual_label = 'llm'
    actual_label = 'human'
    output = insert_df(output, [embedding, actual_label])

  return output

def embed_files_codebert(dir: DataFileDirectory, tokenizer, model) -> pd.DataFrame:
  output = pd.DataFrame(columns=['code_embeddings', 'actual label'])
  for path, name in dir.get_path_to_file_name_mapping().items():
    code = open(path, 'r', encoding='utf-8').read()
    embedding = get_embedding_roBERTa(code, tokenizer, model)

    actual_label = 'human'
    output = insert_df(output, [embedding, actual_label])

  return output

