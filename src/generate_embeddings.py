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
from utils.file_utils import prompt_save_file, EMBEDDING_EXTENSION
from utils.embeddings_gen import get_embedding, get_embedding_roBERTa
from utils.embedding import expand_embeddings_2

torch.cuda.empty_cache()

checkpoint = "Salesforce/codet5p-110m-embedding"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

encoders = ["neulab/codebert-c", "Salesforce/codet5p-110m-embedding"]
def loadCode(filename: str):
  source = pd.read_pickle(filename)
  return source

def generate(input_file: str, encoder: str):
  tokenizer = AutoTokenizer.from_pretrained(encoder, trust_remote_code=True)
  model = AutoModel.from_pretrained(encoder, trust_remote_code=True).to(device)
  print("Loaded Model")

  source = loadCode(input_file)
  print(f"Loaded code. Count: {source.shape}")
  embeddings = source['code'].tolist()
  print(len(embeddings))
  output = source['actual label'].to_frame()

  with alive_bar(len(source)) as bar:
    for index, row in enumerate(embeddings):
      if(encoder == encoders[0]):
        embeddings[index] = get_embedding_roBERTa(row, tokenizer, model)
      else:
        embeddings[index] = get_embedding(row, tokenizer, model, 1024)
      bar()

  output['code_embeddings'] = embeddings
  return output

if __name__ == "__main__":
  for index, model in enumerate(encoders):
    print(f"{index}: {model}")
  choice = int(input("Choose Encoder"))
  chosen_encoder = encoders[choice]

  file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + prepared_dir
  question_files_class = DataFileDirectory(file_path, '.code.pkl')
  data_input_file = question_files_class.get_file("Select file to embed")

  output_map, multiple = prompt_save_file(question_files_class, EMBEDDING_EXTENSION)
  data_output_file = output_map[data_input_file]
  

  print("="*40)
  print("Reading from: ", data_input_file)
  print("Saving to: ", data_output_file)
  print(f"Using {chosen_encoder}")

  output = generate(data_input_file, chosen_encoder)
  
  output.to_pickle(data_output_file)
  print("Saved to ", data_output_file)
