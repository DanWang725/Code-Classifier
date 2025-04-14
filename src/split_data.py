import pandas as pd
import os
from utils.directories import prepared_dir
from utils.file_retrieval import DataFileDirectory
from utils.file_utils import EMBEDDING_EXTENSION, prompt_save_file, get_emb_stats

ratios = [90, 10]

import pandas as pd
import numpy as np

def reproducible_shuffle(df, seed=42):
    # Use the same permutation for all DataFrames of the same length
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True)

if __name__ == "__main__":
   file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + prepared_dir
   question_files_class = DataFileDirectory(file_path, EMBEDDING_EXTENSION, get_emb_stats, {'not_end': ['.train', '.test']})
   
   data_input_file = question_files_class.get_file("Select file to split")

   while data_input_file is not None:
      data_input_file = question_files_class.get_file("Select file to split, exit to continue.")
   
   output_file, _ = prompt_save_file(question_files_class, '.', backup=False)
   for input_file_path, output_file_path in output_file.items():
      print("="*40)
      print("Reading from: ", input_file_path)
      print("Saving splits to: ", output_file_path)

      data = pd.read_pickle(input_file_path)
      data_num = len(data)

      train_split = int(ratios[0]/sum(ratios)*data_num)
      data = reproducible_shuffle(data)

      train = data.iloc[:train_split]
      test = data.iloc[train_split:]

      print(f"Training Split: Size {train.shape[0]}")
      train.to_pickle(output_file_path + "train.emb.pkl")
      print(f"Test Split: Size {test.shape[0]}")
      test.to_pickle(output_file_path + "test.emb.pkl")