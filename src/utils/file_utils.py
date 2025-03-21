from utils.file_retrieval import DataFileDirectory
import os

CODE_EXTENSION = '.code.pkl'
EMBEDDING_EXTENSION = '.emb.pkl'
PROBLEM_EXTENSION = 'pbm.pkl'


def prompt_save_file(file_loader: DataFileDirectory, output_ext: str, backup: bool = True) -> dict[str,str]:
  save_same_name = input("use file names for output? (y/n): ")

  files_to_save = file_loader.get_chosen_files(prefix_path=True)

  if save_same_name == "y":
    output_map = {(x + file_loader.data_ext): x + output_ext for x in files_to_save}
  else:
    output_map = {file_loader.data_path + x + file_loader.data_ext : x for x in file_loader.get_chosen_files()}
    for file_path, file_name in output_map.items():
      print(f"Enter the new name for {file_name}")
      new_file_name = input()
      output_map[file_path] = file_loader.data_path + new_file_name + output_ext

  if(backup):
    for file in output_map.values():
      if(os.path.exists(file)):
        try:
          os.system("cp " + file + " " + file + ".bak")
        except Exception as e:
          print(f"could not backup {file} as a backup already exists.")

  return output_map
