import pandas as pd
import os
import sys
from utils.directories import prepared_dir, ai_dir, human_dir
from utils.file_retrieval import DataFileDirectory
from utils.file_utils import get_code_stats, get_human_code_stats

prompt = "Merging data interface. First select data source: \n1 = AI code\n2 = Human code\nd = Finish\n"

if __name__ == "__main__":
  ai_path = os.path.dirname(os.path.abspath(__file__)) + "/" + ai_dir
  human_path = os.path.dirname(os.path.abspath(__file__)) + "/" + human_dir

  print("Loading data...")
  ai_files = DataFileDirectory(ai_path, '.code.pkl', get_code_stats)
  human_files = DataFileDirectory(human_path, '.code.pkl', get_human_code_stats)

  user_input = input(prompt)
  while user_input != "d":
    if user_input == "1":
      selected_dir = ai_files
      files_choice = ai_files
      file_dir = ai_dir
    elif user_input == "2":
      selected_dir = human_files
      files_choice = human_files
      file_dir = human_dir
    elif user_input == "d":
      break
    else:
      print("Invalid choice. Please try again.")

    selected_dir.get_file("Select Files to Choose")
    user_input = input(prompt)

  
  human_build_df = pd.DataFrame(columns=['identifier','code', 'actual label'])
  ai_build_df = pd.DataFrame(columns=['identifier','code', 'actual label'])

  for file in ai_files.get_chosen_files(prefix_path=True, extension=True):
    data = pd.read_pickle(file)
    ai_build_df = pd.concat([ai_build_df, data], ignore_index=True)

  for file in human_files.get_chosen_files(prefix_path=True, extension=True):
    data = pd.read_pickle(file)
    human_build_df = pd.concat([human_build_df, data], ignore_index=True)      

  human_df = human_build_df["code"].copy()
  ai_df = ai_build_df["code"].copy()

  human_df.columns = ['code']
  ai_df.columns = ['code']

  human_df = human_df.to_frame()
  ai_df = ai_df.to_frame()

  human_df['actual label'] = "human"
  ai_df['actual label'] = "llm"

  merged_df = pd.concat([human_df, ai_df], ignore_index=True).dropna()
  
  output_files = [x[:-9] for x in os.listdir(prepared_dir) if x.endswith(".code.pkl")]

  print("Enter File to Save To. Existing Files to Overwrite (copy will be temporarily saved): ")
  for idx, file in enumerate(output_files):
      print(f"{idx+1}. {file}")
  output_file = input("Filename or Index to Overwrite: ")
  if(output_file.isnumeric() and int(output_file) <= len(output_files)):
      data_output_file = prepared_dir + output_files[int(output_file)-1] + ".code.pkl"
      os.system("cp " + data_output_file + " " + data_output_file + ".old")
  else:
      data_output_file = prepared_dir + output_file + ".code.pkl"
  # Save the merged data
  merged_df.to_pickle(data_output_file)
  print("Data merged and saved successfully.")
