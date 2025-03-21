import pandas as pd

def prepare_data(source: pd.DataFrame, embedding_label: str, target: str):
  data_expanded = expand_embeddings(source, embedding_label)
  return pd.concat([data_expanded, source[[target]]], axis=1)

def expand_embeddings(source: pd.DataFrame, embedding_label: str):
  return pd.DataFrame(source[embedding_label].tolist(), index=source.index)  # Expands each ndarray into separate columns
