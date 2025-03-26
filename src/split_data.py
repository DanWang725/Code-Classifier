import pandas as pd
import os
from utils.directories import prepared_dir
from utils.file_retrieval import DataFileDirectory
from utils.file_utils import EMBEDDING_EXTENSION

ratios = [80, 10, 10]

if __name__ == "__main__":
   file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + prepared_dir
   question_files_class = DataFileDirectory(file_path, EMBEDDING_EXTENSION)
   data_input_file = question_files_class.get_file("Select file to split")

   output_file = input("Enter the file name to save the split data to: ")
   output_file = prepared_dir + output_file + "."
   
   print("="*40)
   print("Reading from: ", data_input_file)
   print("Saving splits to: ", output_file)

   data = pd.read_pickle(data_input_file)
   data_num = len(data)

   train_split = int(ratios[0]/sum(ratios)*data_num)
   val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

   data = data.sample(frac=1, random_state=666)
   train = data.iloc[:train_split]
   dev = data.iloc[train_split:val_split]
   test = data.iloc[val_split:]

   print(f"Training Split: Size {train.shape[0]}")
   train.to_pickle(output_file + "train.emb.pkl")
   print(f"Validation Split: Size {dev.shape[0]}")
   dev.to_pickle(output_file + "dev.emb.pkl")
   print(f"Test Split: Size {test.shape[0]}")
   test.to_pickle(output_file + "test.emb.pkl")