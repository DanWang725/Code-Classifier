import re
from ollama import chat
from ollama import ChatResponse
import pandas as pd
from alive_progress import alive_bar
import json
import os

base_dir = "../../data/ai-code/"

def getCodeFromResponse(content: str):
    matches = re.search(r"```c(.+?)```", content, re.DOTALL)
    if matches is not None:
        return matches.group(1)
    else:
        return None
    

def generateCodeFromChat(question: str):
    response = chat(model='deepseek-r1:8b', messages=[ #14b
    {
        'role': 'system',
        'content': 'Write the C code for the following assignment question'
    },
    {
        'role': 'user',
        'content': question,
    }])
    cleaned_content = (str)(re.sub(r"<think>.*?</think>\n?", "", response.message.content, flags=re.DOTALL))
    return cleaned_content

def generateCodeFromChatWithRetry(question: str, response: str):
    response = chat(model='deepseek-r1-8b-0t', messages=[ #14b
    {
        'role': 'system',
        'content': 'Write the C code for the following assignment question'
    },
    {
        'role': 'user',
        'content': question,
    },
    {
        'role': 'system',
        'content': response,
    },
    {
        'role': 'user',
        'content': "I couldn't find the C code from your last response. Here's the assignment question again, please generate C language code to solve this assignment question." + question + ""
    }])
    cleaned_content = (str)(re.sub(r"<think>.*?</think>\n?", "", response.message.content, flags=re.DOTALL))
    return cleaned_content

def generate(source: pd.DataFrame, data_output_file: str):
    if(os.path.exists(data_output_file)): # If the output file exists, load it
        output = pd.read_pickle(data_output_file)
        outputCode = output['code'].tolist()
        label = output['actual label'].tolist()
    else:
        output = source.loc[:, 'identifier'].to_frame()
        outputCode = [None for x in range(len(source))]
        label = [None for x in range(len(source))]

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

                res = generateCodeFromChat(question)
                code = getCodeFromResponse(res)
                if code is None:
                    print("Could not generate code for " + row['identifier'])
                    label[index] = 'ai-failed'
                    bar()
                    continue
                outputCode[index] = code
                label[index] = 'ai-0.8t'
                bar()
    except Exception as e:
        print(e)

    output['code'] = outputCode
    output['actual label'] = label

    return output

if __name__ == "__main__":
    question_files = [x[:-8] for x in os.listdir(base_dir) if x.endswith(".pbm.pkl")]
    print("Question Files: ")
    for idx, file in enumerate(question_files):
        print(f"{idx+1}. {file}")
    input_file = int(input("Enter the file index to read questions from: "))
    data_input_file = base_dir + question_files[input_file-1] + ".pbm.pkl"

    code_output_files = [x[:-9] for x in os.listdir(base_dir) if x.endswith(".code.pkl")]
    print("Enter File to Save To. Existing Files to Overwrite (copy will be temporarily saved): ")
    for idx, file in enumerate(code_output_files):
        print(f"{idx+1}. {file}")
    output_file = input("Filename or Index to Overwrite: ")
    if(output_file.isnumeric() and int(output_file) <= len(code_output_files)):
        data_output_file = base_dir + code_output_files[int(output_file)-1] + ".code.pkl"
        os.system("cp " + data_output_file + " " + data_output_file + ".old")
    else:
        data_output_file = base_dir + output_file + ".code.pkl"

    print("="*40)
    print("Reading from: ", data_input_file)
    print("Saving to: ", data_output_file)

    df = pd.read_pickle(data_input_file)
    output = generate(df, data_output_file)
    output.to_pickle(data_output_file)