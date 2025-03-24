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
from utils.file_utils import prompt_save_file
from utils.directories import ai_dir
from utils.rate_limit import rate_limits

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_AI")
print(api_key)

if(api_key is not None):
    client = genai.Client(api_key=api_key)

def getCodeFromResponse(content: str):
    matches = re.search(r"```c\n(.+?)```", content, re.DOTALL)
    if matches is not None:
        return matches.group(1)
    else:
        return None
    
@rate_limits(max_calls=10, period=60)
def generateCodeFromGemini(model: str, content: str):
    response = client.models.generate_content(model=model, contents=content)
    return response.text

def generateCodeFromChat(model: str, question: str):
    if(model == 'llama3.1'):
        messages = [
                    {
                        'role': 'system',
                        'content': 'Write only C language code for the given coding question'
                    },
                    {
                        'role': 'user',
                        'content': question,
                    }
                    ]
    elif model == "gemini-2.0-flash":
        messages = "Write only C language code for the given coding question:\n" + question
    else:
        messages = [
                    {
                        'role': 'user',
                        'content': question,
                    },
                        {
                        'role': 'system',
                        'content': 'Write only C language code for the given coding question'
                    }]
        
    if(model == "gemini-2.0-flash"):
        try:
            text_content = generateCodeFromGemini(model, messages)
        except Exception as e:
            print(e)
            sleep(60)
            text_content = generateCodeFromGemini(model, messages)
    else:
        response = chat(model=model, messages=messages)
        text_content = response.message.content

    cleaned_content = (str)(re.sub(r"<think>.*?</think>\n?", "", text_content, flags=re.DOTALL))
    return cleaned_content

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

    try:
        with alive_bar(len(source), dual_line=True) as bar:
            for index, row in source.iterrows():
                question = row['question']
                bar.text("Generating code for " + row['identifier'])
                if label[index] == 'ai':
                    print(row['identifier'], "is not blank")
                    bar()
                    continue
                else:
                    print(row['identifier'], "is blank")

                res = generateCodeFromChat(model, question)
                sys.stderr.write(res)
                code = getCodeFromResponse(res)
                if code is None:
                    print("Could not generate code for " + row['identifier'])
                    label[index] = 'ai-failed'
                    bar()
                    continue
                outputCode[index] = code
                label[index] = 'ai'
                bar()
    except Exception as e:
        print(e)

    output['code'] = outputCode
    output['actual label'] = label

    return output

models = ['deepseek-r1:8b', 'deepseek-r1-8b-0t', 'deepseek-r1:14b', 'deepseek-r1-14b-0t', 'llama3.1', 'gemini-2.0-flash']
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
    question_files_class = DataFileDirectory(file_path, '.pbm.pkl')

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

