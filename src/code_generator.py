import re
from time import sleep
from ollama import chat
from ollama import ChatResponse
import pandas as pd
from alive_progress import alive_bar
import json
import os
import sys
from utils.file_retrieval import DataFileDirectory
from utils.file_utils import prompt_save_file, get_question_stats
from utils.directories import ai_dir
from utils.model_llm import generateCodeFromChat, getCodeFromResponse, models

def generate(source: pd.DataFrame, data_output_file: str, model: str = 'deepseek-r1:8b'):
    if(os.path.exists(data_output_file)): # If the output file exists, load it
        output = pd.read_pickle(data_output_file)
        outputCode = output['code'].tolist()
        label = output['actual label'].tolist()
        if os.path.exists(data_output_file + ".bak"):
            os.remove(data_output_file + ".bak")
        os.rename(data_output_file, data_output_file + ".bak")
    else:
        output = source.loc[:, 'identifier'].to_frame()
        outputCode = [None for x in range(len(source))]
        label = [None for x in range(len(source))]

    print("Size:" , len(source))
    skipped = 0
    failed = 0

    try:
        with alive_bar(len(source), dual_line=True) as bar:
            for index, row in source.iterrows():
                question = row['question']
                bar.text("Generating code for " + row['identifier'])
                if label[index] == 'ai':
                    # print(row['identifier'], "is not blank")
                    bar()
                    skipped = skipped + 1
                    continue

                res = generateCodeFromChat(model, question)
                # sys.stderr.write(res)
                code = getCodeFromResponse(res)
                if code is None:
                    # print("Could not generate code for " + row['identifier'])
                    failed = failed + 1
                    label[index] = 'ai-failed'
                    bar()
                    continue
                outputCode[index] = code
                label[index] = 'ai'
                bar()
    except Exception as e:
        print(e)
    print(f"Generated {len(source) - skipped}, Failed {failed}")
    output['code'] = outputCode
    output['actual label'] = label

    return output

output_extension = ".code.pkl"
input_extension = ".pbl.pkl"

def generation_loop(output_map: dict[str,str], model):
    for data_input_file, data_output_file in output_map.items():
        print("="*40)
        print("Reading from: ", data_input_file)
        print("Saving to: ", data_output_file)
        print("Using model: ", model)

        df = pd.read_pickle(data_input_file)
        output = generate(df, data_output_file, model)
        output.to_pickle(data_output_file)

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + ai_dir
    question_files_class = DataFileDirectory(file_path, '.pbm.pkl', get_question_stats)

    file = ""
    while file is not None:
        file = question_files_class.get_file("Choose files to generate questions for, exit to continue. ")

    output_map, multiple = prompt_save_file(question_files_class, output_extension, False, True)

    for idx, model in enumerate(models):
        print(f"{idx+1}. {model}")
    model_input = int(input("Enter the model index to use: "))
    model = models[model_input-1]

    if (multiple):
        for file_map in output_map:
            generation_loop(file_map, model)
    else:
        generation_loop(output_map=output_map, model=model)

