import os
import re
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
JWT_TOKEN = os.getenv("JWT_TOKEN")

# API endpoint (using the working endpoint from your curl example)
API_ENDPOINT = "http://localhost:3000/api/chat/completions"

def call_deepseek(prompt):
    """
    Sends a prompt to the DeepSeek API and returns the model's answer text.
    """
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-r1:32b",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None

    data = response.json()
    try:
        # Using the structure provided:
        # {'choices': [{'message': {'content': ...}, ...}]}
        answer_text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print("Error parsing response:", e)
        return None
    return answer_text


def split_question_into_subqueries(question: str, num_subqueries: int = 2) -> list:
    """
    Uses DeepSeek to generate subqueries from a given question.
    """
    prompt = (
        f"Given a question, split it into exactly {num_subqueries} smaller questions that need to be answered "
        "in order to answer the overall question. Do not give an answer to the orignial question, just sub-questions that need to be answered\n\n"
        f"Question: '{question}'\nOutput:"
    )

    answer_text = call_deepseek(prompt)
    if not answer_text:
        return []

    subqueries = re.findall(r"\d+\.\s*(.*)", answer_text) #regex to match a numbered list format
    return subqueries[:num_subqueries] if subqueries else answer_text.split("\n")[:num_subqueries]


def main():
    #Experiment subqueries:
    question = "Can a suit of armor conduct electricity?"
    subqueries = split_question_into_subqueries(question, num_subqueries=3)
    print("Generated Subqueries:", subqueries)

if __name__ == "__main__":
    main()
