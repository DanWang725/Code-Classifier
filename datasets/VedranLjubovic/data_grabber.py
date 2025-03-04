from ollama import chat
from ollama import ChatResponse
import os
import sys
import re
import pandas as pd

question_generation_prompt = """You will be given c langauge code from a file. Your job will be to generate a first-year university computer science course assignment question that the student would have written the code for.

Write the assignment question in clear, structured English, formatted into paragraphs, providing clear learning outcomes."""

english_enforce = """Always respond in English, regardless of the language the user speaks."""
question_verification_prompt = "Verify that the following assignment question compares to one seen in a first year computer science course, and is written in only English, not including any variable or function names. Write either 'ANSWERYES' or 'ANSWERNO'. Verify the following text:"
question_verification_prompt_2 = "Is this text in English words and have the an assignment title? Respond with 'ANSWERYES' or 'ANSWERNO'. Here is the text:"
summarization_prompt = """Summarize the following assignment question into a single paragraph. The summary should be concise and capture what type of program needs to be written, and what problem it is solving. Do not include any new information in the summary. """


def verifyResponse(question: str):
  if(question.find("Assignment") != -1):
    return "ANSWERYES"
  else:
    return "ANSWERNO"
#figure out how to format stuff
def retrieveResponse(prompt: str, question: str): 
    response = chat(model='llama3.1', messages=[ #14b
        {
            'role': 'system',
            'content': english_enforce,
        },
        {
            'role': 'system',
            'content': prompt,
        },
        {
            'role': 'user',
            'content': question,
        }
    ])
    # sys.stderr.write(response.message.content)
    cleaned_content = (str)(re.sub(r"<think>.*?</think>\n?", "", response.message.content, flags=re.DOTALL))
    return cleaned_content 

def retrieveEnglishRetry(prompt: str, question: str, response: str):
  response = chat(model='llama3.1', messages=[ #14b
      {
          'role': 'system',
          'content': english_enforce,
      },
      {
          'role': 'system',
          'content': prompt,
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
          'content': "Here's the code again: " + question + "Last time, you gave me the assignment that wasn't in english. Just a reminder, I want you to generate an assignment question that the code was written for. Please give me assignment in english."
      }
  ])
  # sys.stderr.write(response.message.content)
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
  questionIdentifier = re.search(r'([A-Z]\d-[A-Z]\d-\d+)', sys.argv[1])[1]
  questions = open_pickle("questions.pkl", ["question", "identifier"])

  if(len(sys.argv) != 2):
    print("Usage: python data_grabber.py <input_file>")
    sys.exit(1)
  file = open(sys.argv[1], "r")
  code = "\n".join(file.readlines())  
  file.close()

  print("Generating Problem Statement")
  if(questions.loc[questions['identifier'] == questionIdentifier].shape[0] > 0):
    print("Already Generated")
    question_response = questions.loc[questions['identifier'] == questionIdentifier].iloc[0]['question']
  else:
    question_response = retrieveResponse(question_generation_prompt, code)

  print("Verifying Question")
  retryCount = 0
  verification_response = verifyResponse(question_response)
  while(verification_response.find("ANSWERYES") == -1):
    print("Failed Verification: " + question_response[0:100])
    retryCount += 1
    if(retryCount >= 3):
      sys.stderr.write(questionIdentifier + " RETRIED TOO MANY TIMES \n")
      sys.exit(1)
    
    sys.stderr.write(questionIdentifier + " RETRY " + str(retryCount) + "\n")
    print("Retry #" + str(retryCount) + ": Generating Problem Statement")
    question_response = retrieveEnglishRetry(question_generation_prompt, code, question_response)
    print("Verifying Question")
    verification_response = verifyResponse(question_response)

  print("Passed Verification: " + question_response[0:100])

  print("Generating Summary")
  summary_response = retrieveResponse(summarization_prompt, question_response)
  print("Saving....")
  questions = insert_df(questions, [question_response, questionIdentifier])
  # questions.to_pickle("questions.pkl")
  print(summary_response)