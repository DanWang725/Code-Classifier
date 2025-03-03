from transformers import AutoModel, AutoTokenizer
import pandas as pd
tqdm.pandas()


files = "data/processed.pkl"

outputFile = "data/embeddings.pkl"


checkpoint = "Salesforce/codet5p-110m-embedding"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

def loadCode(filename: str):
  source = pd.read_pickle(filename)
  return source

def main():
  
  tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
  model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
  print("Loaded Model")
  source = loadCode(files)
  print(f"Loaded code. Count: {source.shape}")
  source['embeddings'] = source['code'].progress_apply(lambda x: model(tokenizer.encode(x, return_tensors="pt").to(device))[0])
  source.to_pickle(outputFile)



# inputs = tokenizer.encode("def print_hello_world():\tprint('Hello World!')", return_tensors="pt").to(device)
# embedding = model(inputs)[0]