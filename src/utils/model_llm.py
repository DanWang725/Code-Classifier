import re
from time import sleep
from ollama import chat
from ollama import ChatResponse
import pandas as pd
from alive_progress import alive_bar
import json
import os
import sys

from utils.rate_limit import rate_limits

from dotenv import load_dotenv
from google import genai

models = ['deepseek-r1:8b', 'deepseek-r1-8b-0t', 'deepseek-r1:14b', 'deepseek-r1-14b-0t', 'llama3.1', 'gemini-2.0-flash']

load_dotenv()

api_key = os.getenv("GOOGLE_AI")

if(api_key is not None):
    client = genai.Client(api_key=api_key)

def getCodeFromResponse(content: str):
    matches = re.search(r"```c\n(.+?)```", content, re.DOTALL)
    if matches is not None:
        return matches.group(1)
    else:
        return None

@rate_limits(max_calls=10, period=60)
def generateFromGemini(content: str, model: str = "gemini-2.0-flash"):
    response = client.models.generate_content(model=model, contents=content)
    if(response is None):
        print("FAILURE:", content)
    return response.text

def llamaChat(question:str):
    response = chat(model='llama3.1', messages=[{'role': 'user', 'content': question}])
    return response.message.content

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
            text_content = generateFromGemini(messages, model)
        except Exception as e:
            print(e)
            sleep(60)
            text_content = generateFromGemini(messages, model)
    else:
        response = chat(model=model, messages=messages)
        text_content = response.message.content

    cleaned_content = (str)(re.sub(r"<think>.*?</think>\n?", "", text_content, flags=re.DOTALL))
    return cleaned_content