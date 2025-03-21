import pandas as pd
import os
import sys

base_dir = "../../data/"
ai_code_dir = base_dir + "ai-code/"
human_code_dir = base_dir + "human-written/"
output_code_dir = base_dir + "prepared/"

prompt = "Merging data interface. First select data source: \n1 = AI code\n2 = Human code\nd = Finish\n"

if __name__ == "__main__":
  ai_files = [x[:-9] for x in os.listdir(ai_code_dir) if x.endswith(".code.pkl")]
  human_files = [x[:-9] for x in os.listdir(human_code_dir) if x.endswith(".code.pkl")]

  added_files = []
  human_dir = pd.DataFrame(columns=['identifier','code', 'actual label'])
  ai_dir = pd.DataFrame(columns=['identifier','code', 'actual label'])

  user_input = input(prompt)
  while user_input != "d":
    if user_input == "1":
      files_choice = ai_files
      file_dir = ai_code_dir
    elif user_input == "2":
      files_choice = human_files
      file_dir = human_code_dir
    elif user_input == "d":
      break
    else:
      print("Invalid choice. Please try again.")
  
    for idx, file in enumerate(files_choice):
      print(f"{idx+1}. {file}")
    input_file = int(input("Enter the file number to merge in: "))

    added_files.append(files_choice[input_file-1])

    data_input_file = file_dir + "/" + files_choice[input_file-1] + ".code.pkl"
    data = pd.read_pickle(data_input_file)
    if file_dir == ai_code_dir:
      ai_dir = pd.concat([ai_dir, data], ignore_index=True)
    else:
      human_dir = pd.concat([human_dir, data], ignore_index=True)
    
    print("Data currently added: ", added_files)
    user_input = input(prompt)

  human_df = human_dir["code"].copy()
  ai_df = ai_dir["code"].copy()

  human_df.columns = ['code']
  ai_df.columns = ['code']

  human_df = human_df.to_frame()
  ai_df = ai_df.to_frame()

  human_df['actual label'] = "human"
  ai_df['actual label'] = "llm"

  merged_df = pd.concat([human_df, ai_df], ignore_index=True).dropna()
  
  output_files = [x[:-9] for x in os.listdir(output_code_dir) if x.endswith(".code.pkl")]

  print("Enter File to Save To. Existing Files to Overwrite (copy will be temporarily saved): ")
  for idx, file in enumerate(output_files):
      print(f"{idx+1}. {file}")
  output_file = input("Filename or Index to Overwrite: ")
  if(output_file.isnumeric() and int(output_file) <= len(output_files)):
      data_output_file = output_code_dir + output_files[int(output_file)-1] + ".code.pkl"
      os.system("cp " + data_output_file + " " + data_output_file + ".old")
  else:
      data_output_file = output_code_dir + output_file + ".code.pkl"
  # Save the merged data
  merged_df.to_pickle(data_output_file)
  print("Data merged and saved successfully.")
