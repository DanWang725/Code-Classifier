import pandas as pd

ENCODERS = ["neulab/codebert-c", "Salesforce/codet5p-110m-embedding", "bigcode/starcoder"]
EncoderMap = {ENCODERS[0]: 'codebert', ENCODERS[1]: 'codet5p', ENCODERS[2]: 'starcoder'}

def prepare_data(source: pd.DataFrame, embedding_label: str, target: str):
  data_expanded = expand_embeddings(source, embedding_label)
  return pd.concat([data_expanded, source[[target]]], axis=1)

def expand_embeddings(source: pd.DataFrame, embedding_label: str):
  processed = source[embedding_label].apply(lambda x: x.flatten())
  return pd.DataFrame(processed.tolist(), index=source.index)  # Expands each ndarray into separate columns

def expand_embeddings_2(source: pd.DataFrame, embedding_label: str):
  source[embedding_label] = source[embedding_label].apply(lambda x: x.flatten().tolist() if x is not None else None)
  return source

def insert_df(df: pd.DataFrame, row: list): 
  df.loc[-1] = row
  df.index = df.index + 1  # shifting index
  df = df.sort_index()  # sorting by index
  return df