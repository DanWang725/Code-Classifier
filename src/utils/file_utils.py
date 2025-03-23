from typing import Union
from utils.file_retrieval import DataFileDirectory
import os

CODE_EXTENSION = '.code.pkl'
EMBEDDING_EXTENSION = '.emb.pkl'
PROBLEM_EXTENSION = 'pbm.pkl'


def prompt_save_file(file_loader: DataFileDirectory, output_ext: str, backup: bool = True, allow_multiple: bool = False) -> tuple[list[dict[str, str]] | dict[str, str], bool]:
  if(allow_multiple):
    multiple = input('Copies to generate: (default = 1): ')
  else:
    multiple = "nope"
  
  save_same_name = input("use file names for output? (y/n): ")

  files_to_save = file_loader.get_chosen_files(prefix_path=True)

  if save_same_name == "y":
    if (multiple.isdigit() and int(multiple) > 1):
      output_map = [{(x + file_loader.data_ext): f"{x}-{i + 1}"+ output_ext for x in files_to_save} for i in range(int(multiple))]
    else:
      output_map = {(x + file_loader.data_ext): x + output_ext for x in files_to_save}
  else:
    temp_output_map = {file_loader.data_path + x + file_loader.data_ext : x for x in file_loader.get_chosen_files()}
    for file_path, file_name in temp_output_map.items():
      print(f"Enter the new name for {file_name}")
      new_file_name = input()
      temp_output_map[file_path] = file_loader.data_path + new_file_name
    
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
