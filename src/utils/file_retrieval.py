import os
import sys
import pandas as pd
import re
      # self.chosen_files[x] is False and
      # (self.settings['start'] is None or re.match(self.settings['start'], x)) and
      # (self.settings['end'] is None or re.search(self.settings['end'] + r'$', x)) and
      # (self.settings['contains'] is None or x.find(self.settings['contains']) != -1)]
def starts_with(string: str, start: str) -> bool:
  return re.match(start, string) is not None
def ends_with(string: str, end: str) -> bool:
  return re.search(end + r'$', string) is not None
def contains(string: str, substring: str) -> bool:
  return string.find(substring) != -1


settings_map = {
  "start": starts_with,
  "end": ends_with,
  "contains": contains,
  "not_start": lambda string, start: not starts_with(string, start),
  "not_end": lambda string, end: not ends_with(string, end),
  "not_contains": lambda string, substring: not contains(string, substring),
}

allowed_multi = ["not_start", "not_end", "not_contains"]
class DataFileDirectory:
  files: list[str]
  chosen_files: dict[str, bool]
  data_path: str
  data_ext: str
  settings: dict[str, str | list[str]]
  
  def __init__(self, data_dir: str, extension: str, stat_func = None, initial_settings: dict[str, str | list[str]] = None):
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
    self.settings = {}
    if initial_settings is not None:
      for key, setting in initial_settings.items():
        if key in settings_map:
          if isinstance(setting, str):
            self.settings[key] = setting
          elif isinstance(setting, list) and key in allowed_multi:
            self.settings[key] = setting
          else:
            raise ValueError(f"Invalid setting value for {key}: {setting}")
        else:
          raise ValueError(f"Invalid setting key: {key}")

  def print_edit_settings(self):
    print("Settings:")
    print("'start?<start-string>' - filters results to starting substring")
    print("'end?<end-string>' - filters results to ending substring")
    print("'contains?<substring>' - filters results to containing substring")
    print("Enter the setting to change, or type 'exit' to leave without making changes")
  
  def set_settings(self, choice):
    setting, value = choice.split("?")
    if setting == "exit":
      return
    if setting in allowed_multi:
      if(value == '' or value is None):
        self.settings[setting] = []
      else:
        if setting not in self.settings:
          self.settings[setting] = [value]
        else:
          self.settings[setting].append(value)
    else:
      self.settings[setting] = value

  def _get_selectable_files(self) -> list[str]:
    selectable_files = [x for x in self.chosen_files if self.chosen_files[x] is False ]
    selectable_files2 = [x for x in selectable_files if all(settings_map[setting_key](x, self.settings[setting_key]) for setting_key in self.settings.keys() if setting_key not in allowed_multi)]
    selectable_files3 = [x for x in selectable_files2 if all(all(settings_map[setting_key](x, setting_value) for setting_value in self.settings[setting_key]) for setting_key in self.settings.keys() if setting_key in allowed_multi)]
    return selectable_files3
  
  def _get_selected_files(self) -> list[str]:
    return [x for x in self.chosen_files if self.chosen_files[x] is True]

  def get_file(self, message: str, contains: str = None) -> str | None:
    if contains is not None:
      self.settings['contains'] = contains
    chosen_file = ""
    while chosen_file == "":
      valid_files = self._get_selectable_files()
      print(str(self._get_selected_files()))
      if(sys.stdout.isatty()):  
        print(message + "Enter the number or file name.\ntype 'help' for settings, 'exit' to exit.")
        print(f"Settings | {' '.join([f'{key}: {self.settings[key]}' for key in self.settings])}")
        for idx, file in enumerate(valid_files):
          print(f"{idx}. {(self.stats[file] if self.stats is not None else '')} - {file}")

      choice = input()
      if(choice == "help"): 
        self.print_edit_settings()
        continue
        
      if(choice.split("?")[0] in settings_map):
        self.set_settings(choice)
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