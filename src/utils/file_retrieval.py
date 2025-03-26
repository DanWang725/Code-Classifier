import os
import sys
import pandas as pd
import re
class DataFileDirectory:
  files: list[str]
  chosen_files: dict[str, bool]
  data_path: str
  data_ext: str
  settings: dict[str, str]
  
  def __init__(self, data_dir: str, extension: str, stat_func = None):
    self.files = [x[:-len(extension)] for x in os.listdir(data_dir) if x.endswith(extension)]
    self.chosen_files = {x: False for x in self.files}
    if stat_func is not None:
      self.stats = {}
      for file in self.files:
        contents = pd.read_pickle(data_dir + file + extension)
        self.stats[file] = "\t"+stat_func(contents)
    else:
      self.stats = None

    self.data_path = data_dir
    self.data_ext = extension
    self.settings = {
      "start": None,
      "end": None,
      "contains": None,
    }
  

  def print_edit_settings(self):
    print("Settings:")
    print("'start?<start-string>' - filters results to starting substring")
    print("'end?<end-string>' - filters results to ending substring")
    print("'contains?<substring>' - filters results to containing substring")
    print("Enter the setting to change, or type 'exit' to leave without making changes")
  
  def set_settings(self):
    self.print_edit_settings()
    choice = input()
    setting, value = choice.split("?")
    if setting == "exit":
      return
    if setting in self.settings:
      self.settings[setting] = value
    else:
      print("invalid setting: " + setting)

  def _get_selectable_files(self) -> list[str]:
    return [x for x in self.chosen_files if 
      self.chosen_files[x] is False and
      (self.settings['start'] is None or re.match(self.settings['start'], x)) and
      (self.settings['end'] is None or re.search(self.settings['end'] + r'$', x)) and
      (self.settings['contains'] is None or x.startswith(self.settings['contains']))]
  
  def _get_selected_files(self) -> list[str]:
    return [x for x in self.chosen_files if self.chosen_files[x] is True]

  def get_file(self, message: str) -> str | None:
    chosen_file = ""
    while chosen_file == "":
      valid_files = self._get_selectable_files()
      print(str(self._get_selected_files()))
      if(sys.stdout.isatty()):  
        print(message + "Enter the number or file name.\ntype 'help' for settings, 'exit' to exit.")
        print(f"Settings | start: {self.settings['start']} | end {self.settings['end']} | contains {self.settings['contains']}")
        for idx, file in enumerate(valid_files):
          print(f"{idx}. {(self.stats[file] if self.stats is not None else '')} - {file}")

      choice = input()
      if(choice == "help"): 
        self.set_settings()
        continue
      
      if(choice.isdigit() and int(choice) < len(valid_files)):
        chosen_file = self.data_path + valid_files[int(choice)] + self.data_ext
        self.chosen_files[valid_files[int(choice)]] = True
      elif choice == "exit":
        chosen_file = None
      elif choice == "-a":
        chosen_file = valid_files[0]
        for valid_file in valid_files:
          self.chosen_files[valid_file] = True
      elif choice in valid_files:
        chosen_file = self.data_path + choice + self.data_ext
        self.chosen_files[choice] = True
      else:
        print("invalid input")
    return chosen_file

  def get_chosen_files(self, prefix_path: bool = False, extension: bool = False) -> list[str]:
    base = [(self.data_path if prefix_path else '') + x + (self.data_ext if extension else '') for x in self.chosen_files if self.chosen_files[x] is True]
    return base
  
  def get_path_to_file_name_mapping(self):
    return {(self.data_path + x + self.data_ext): x for x in self.files if self.chosen_files[x] is True}