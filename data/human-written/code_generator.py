import re
from ollama import chat
from ollama import ChatResponse
import pandas as pd
from alive_progress import alive_bar
import json
import os

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

def generate(source: pd.DataFrame):
    if(os.path.exists("output.csv")):
        output = pd.read_csv("output-2.csv")
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

    output.to_csv("output-2.csv", index=False)

if __name__ == "__main__":
    df = pd.read_pickle("questions.pkl")
    generate(df)