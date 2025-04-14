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
from utils.embedding import expand_embeddings_2, ENCODERS, EncoderMap

torch.cuda.empty_cache()

device = "cuda"  # for GPU usage or "cpu" for CPU usage

def loadCode(filename: str):
  source = pd.read_pickle(filename)
  return source

def generate(tokenizer, model, input_file: str, encoderStr: str, embedding_length: int = 512):
  source = loadCode(input_file)
  print(f"Loaded code. Count: {source.shape}")
  embeddings = source['code'].tolist()
  print(len(embeddings))
  output = source['actual label'].to_frame()

  with alive_bar(len(source)) as bar:
    for index, row in enumerate(embeddings):
      if(encoderStr == ENCODERS[0]):
        embeddings[index] = get_embedding_roBERTa(row, tokenizer, model, embedding_length)
      else:
        embeddings[index] = get_embedding(row, tokenizer, model, embedding_length)
      bar()

  output['code_embeddings'] = embeddings
  prev_shape = output.shape[0]
  output = output[output['code_embeddings'].notna()]
  print(f"Filtered code. Removed: {prev_shape - output.shape[0]} entries")
  return output

if __name__ == "__main__":
  for index, model in enumerate(ENCODERS):
    print(f"{index}: {model}")
  choice = int(input("Choose Encoder"))
  chosen_encoder = ENCODERS[choice]

  tokenizer = AutoTokenizer.from_pretrained(chosen_encoder, trust_remote_code=True)
  model = AutoModel.from_pretrained(chosen_encoder, trust_remote_code=True).to(device)
  print("Loaded Model")

  file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + prepared_dir
  question_files_class = DataFileDirectory(file_path, '.code.pkl')
  data_input_file = question_files_class.get_file("Select file to embed")
  while data_input_file is not None:
    data_input_file = question_files_class.get_file("Select file to embed, exit to continue.")

  output_map, multiple = prompt_save_file(question_files_class, '-')
  emb_lengths = []
  ans = 10
  while len(emb_lengths) == 0 or ans > 0:
    ans = int(input(f"Enter the embedding length {emb_lengths} -1 to exit: "))
    if ans > 0:
      emb_lengths.append(ans)

  for data_input_file, data_output_file in output_map.items():
    for emb_length in emb_lengths:
      embedding_length = emb_length
      choice = 'y'
      if len(emb_lengths) == 1:
        choice = input("Show embedding length in output file name? (y/n)")
      
      if choice != 'y':
        data_output_file_path = data_output_file + EncoderMap[chosen_encoder] + EMBEDDING_EXTENSION
      else:
        data_output_file_path = data_output_file + str(embedding_length) + "-" + EncoderMap[chosen_encoder] + EMBEDDING_EXTENSION

      print("="*40)
      print("Reading from: ", data_input_file)
      print("Saving to: ", data_output_file_path)
      print(f"Using {chosen_encoder}")

      output = generate(tokenizer, model, data_input_file, chosen_encoder, embedding_length=embedding_length)
      
      output.to_pickle(data_output_file_path)
      print("Saved to ", data_output_file_path)
