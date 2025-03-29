from typing import Union
from utils.file_retrieval import DataFileDirectory
import os
import pandas as pd

CODE_EXTENSION = '.code.pkl'
EMBEDDING_EXTENSION = '.emb.pkl'
PROBLEM_EXTENSION = 'pbm.pkl'


def prompt_save_file(file_loader: DataFileDirectory, output_ext: str, backup: bool = True, allow_multiple: bool = False, no_directory: bool = False) -> tuple[list[dict[str, str]] | dict[str, str], bool]:
  if(allow_multiple):
    multiple = input('Copies to generate: (default = 1): ')
  else:
    multiple = "nope"
  
  save_same_name = input("use file names for output? (y/n): ")

  files_to_save = file_loader.get_chosen_files()

  if save_same_name == "y":
    if (multiple.isdigit() and int(multiple) > 1):
      output_map = [{(file_loader.data_path + x + file_loader.data_ext): (file_loader.data_path if not no_directory else "") + f"{x}-{i + 1}"+ output_ext for x in files_to_save} for i in range(int(multiple))]
    else:
      output_map = {(file_loader.data_path + x + file_loader.data_ext): (file_loader.data_path if not no_directory else "") + x + output_ext for x in files_to_save}
  else:
    temp_output_map = {file_loader.data_path + x + file_loader.data_ext : x for x in file_loader.get_chosen_files()}
    for file_path, file_name in temp_output_map.items():
      print(f"Enter the new name for {file_name}")
      new_file_name = input()
      temp_output_map[file_path] = (file_loader.data_path if not no_directory else "") + new_file_name
    
    if (multiple.isdigit() and int(multiple) > 1):
      output_map = [{input_file: f"{output_file}-{i + 1}"+ output_ext for input_file, output_file in temp_output_map.items()} for i in range(int(multiple))]
    else:
      output_map = {input_file: output_file + output_ext for input_file, output_file in temp_output_map.items()}


  if(backup and not multiple):
    for file in output_map.values():
      if(os.path.exists(file)):
        try:
          os.system("cp " + file + " " + file + ".bak")
        except Exception as e:
          print(f"could not backup {file} as a backup already exists.")
  
  multiple_flag = multiple.isdigit() and int(multiple) > 1
  return output_map, multiple_flag

def get_label_stats(source: pd.DataFrame, col_label: str, labels: list[str]):
  try:
    return {label: len(source.loc[(source[col_label] == label)][col_label].tolist()) for label in labels}
  except Exception as e:
    print('wrong label: ' + col_label)
    if(col_label == 'label'):
      col_label = 'actual label'
      return {label: len(source.loc[(source[col_label] == label)][col_label].tolist()) for label in labels}
    return e
  
def get_col_stats(source: pd.DataFrame, col_label: str):
  try: 
    return len(source[col_label].tolist())
  except Exception as e:
    print(e)
    return 1
  
def get_emb_stats(source: pd.DataFrame):
  stats = get_label_stats(source, "actual label", ['llm', 'human'])
  return f"[Human: {stats['human']}, AI: {stats['llm']}]"

def get_code_stats(source: pd.DataFrame):
  stats = get_label_stats(source, "actual label", ['ai', 'human', 'ai-failed'])
  return f"[H: {stats['human']}, AI: {stats['ai']}, AI-f: {stats['ai-failed']}]"

def get_human_code_stats(source: pd.DataFrame):
  stats = get_label_stats(source, "label", ['ai', 'human', 'ai-failed'])
  return f"[H: {stats['human']}, AI: {stats['ai']}, f(AI): {stats['ai-failed']}]"

def get_question_stats(source: pd.DataFrame):
  stats = get_col_stats(source, "question")
  return f"[Questions: {stats}]"