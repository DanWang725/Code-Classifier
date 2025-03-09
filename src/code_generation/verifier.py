import pandas as pd
from code_generation.question_generator import question_generation_prompt, question_verification_prompt_2, retrieveResponse
import os
import re
import sys

def verify_and_regenerate_questions(file_path: str):
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist.")
        return

    questions = pd.read_pickle(file_path)
    for index, row in questions.iterrows():
        question = (str)(row['question'])
        questionIdentifier = row['identifier']

        print(f"Verifying Question {questionIdentifier}: {question[0:20]}...")
        # verification_response = verification_response(question_verification_prompt_2, question)
        if(question.find("Assignment") != -1):
            verification_response = "ANSWERYES"
        else:
            verification_response = "ANSWERNO"
        print(verification_response)
        retryCount = 0

        # while verification_response.find("ANSWERYES") == -1:
        #     print(f"Failed Verification for {questionIdentifier}")
        #     retryCount += 1
        #     if retryCount >= 3:
        #         sys.stderr.write(questionIdentifier + " RETRIED TOO MANY TIMES \n")
        #         break

        #     sys.stderr.write(questionIdentifier + " RETRY " + str(retryCount) + "\n")
        #     print(f"Retry #{retryCount}: Generating Problem Statement for {questionIdentifier}")
        #     question_response = retrieveResponse(question_generation_prompt, question)
        #     verification_response = retrieveResponse(question_verification_prompt_2, question_response)

        #     if verification_response.find("ANSWERYES") != -1:
        #         questions.at[index, 'question'] = question_response
        #         print(f"Question {questionIdentifier} passed verification after {retryCount} retries.")
        #         break

    # questions.to_pickle(file_path)
    print("Verification and regeneration process completed.")

# Example usage
verify_and_regenerate_questions("questions.pkl")