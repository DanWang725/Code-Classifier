from ollama import chat
from ollama import ChatResponse
import os
import sys
import re
import pandas as pd

question_generation_prompt = """You will be given c langauge code from a file. Your job will be to generate a first-year university computer science course assignment question that the student would have written the code for. 

Write the assignment question in clear, structured English, formatted into paragraphs, providing clear learning outcomes."""

question_verification_prompt = "Verify that the following assignment question compares to one seen in a first year computer science course. If it is clear, include 'ANSWERYES' in your response. Otherwise include 'ANSWERNO' in your response."

summarization_prompt = """Summarize the following assignment question into a single paragraph. The summary should be concise and capture what type of program needs to be written, and what problem it is solving. Do not include any new information in the summary. """

#figure out how to format stuff
def retrieveResponse(prompt: str, question: str): 
    response = chat(model='llama3.1', messages=[ #14b
        {
            'role': 'user',
            'content': prompt + "\n\n" + question,
        }
    ])
    sys.stderr.write(response.message.content)
    cleaned_content = (str)(re.sub(r"<think>.*?</think>\n?", "", response.message.content, flags=re.DOTALL))
    return cleaned_content 

def insert_df(df: pd.DataFrame, row: list):
  df.loc[-1] = row
  df.index = df.index + 1  # shifting index
  df = df.sort_index()  # sorting by index
  return df

def open_pickle(file_path: str, columns: list):
    if(os.path.exists(file_path)):
      return pd.read_pickle(file_path)
    else:
      return pd.DataFrame(columns=columns)

if __name__ == '__main__':
  questionIdentifier = re.search(r'([A-Z]\d-[A-Z]\d)', sys.argv[1])[1]
  questions = open_pickle("questions.pkl", ["question", "identifier"])
  # codeOutput = open_pickle("programs.pkl")

  if(len(sys.argv) != 2):
    print("Usage: python data_grabber.py <input_file>")
    sys.exit(1)
  file = open(sys.argv[1], "r")
  code = "\n".join(file.readlines())  
  file.close()
  print("Generating Problem Statement")
  question_response = retrieveResponse(question_generation_prompt, code)

  print("Verifying Question")
  verification_response = retrieveResponse(question_verification_prompt, question_response)
  print(verification_response)
  if(verification_response.find("ANSWERYES") == -1):
    print(question_response)
    print("Question is not clear. Exiting....")
    sys.exit(1)
  else:
    print("Question is clear")
  print("Generating Summary")
  summary_response = retrieveResponse(summarization_prompt, question_response)
  print("Saving....")
  questions = insert_df(questions, [question_response, questionIdentifier])
  # codeOutput = codeOutput.append({"code": code, "identifier": questionIdentifier}, ignore_index=True)
  questions.to_pickle("questions.pkl")
  # codeOutput.to_pickle("programs.pkl")
  print(summary_response)