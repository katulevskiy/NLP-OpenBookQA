import os
import json
import re
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def extract_final_answer(answer_text):
    """
    Extracts the final answer label from the model's answer.
    It looks for a line in the answer like: "Final Answer: <label>"
    """
    match = re.search(r"Final Answer:\s*([A-D])", answer_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def process_question(data, use_context, context_text):
    """
    Processes a single question:
      - Builds the prompt (with or without additional context)
      - Calls DeepSeek
      - Extracts the final answer label.
    Returns a dict with all relevant information.
    """
    qid = data.get("id", "unknown")
    question_stem = data["question"]["stem"]
    choices = data["question"]["choices"]
    answer_key = data["answerKey"]

    # Build the prompt
    prompt = ""
    if use_context:
        prompt += f"Context:\n{context_text}\n\n"
    prompt += f"Question: {question_stem}\n"
    prompt += "Choices:\n"
    for choice in choices:
        prompt += f"{choice['label']}: {choice['text']}\n"
    prompt += (
        "\nPlease choose the correct answer label (A, B, C, or D) and "
        "state your final decision at the end of your answer in the format 'Final Answer: <label>'."
    )

    answer_text = call_deepseek(prompt)
    final_label = extract_final_answer(answer_text) if answer_text else None
    correct = 1 if final_label == answer_key else 0

    return {
        "qid": qid,
        "answer_key": answer_key,
        "answer_text": answer_text,
        "final_label": final_label,
        "correct": correct,
    }


def run_experiment(questions_num, use_context=False, context_text=""):
    """
    Processes questions concurrently from dev.jsonl, prints the expected answer and
    full DeepSeek response, and computes the accuracy.
    """
    correct = 0
    total = 0
    dev_file = os.path.join("OpenBookQA-V1-Sep2018", "Data", "Main", "dev.jsonl")

    with open(dev_file, "r") as f:
        # Process only the first questions_num questions
        lines = list(islice(f, questions_num))

    # Parse each non-empty line as JSON
    questions = [json.loads(line) for line in lines if line.strip()]

    results = []
    # Adjust max_workers as needed (here using 5 workers)
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {
            executor.submit(process_question, data, use_context, context_text): data
            for data in questions
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing questions"
        ):
            result = future.result()
            results.append(result)
            print(f"\n--- Question {result['qid']} ---")
            print(f"Expected Answer: {result['answer_key']}")
            if result["answer_text"] is None:
                print("DeepSeek Response: No answer received.")
            else:
                print("DeepSeek Response:")
                print(result["answer_text"])
            if result["final_label"] is None:
                print("Could not extract final answer label from DeepSeek response.\n")
            else:
                print(f"Extracted Final Answer Label: {result['final_label']}\n")

    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = (correct / total * 100) if total > 0 else 0
    context_str = "with context" if use_context else "without context"
    print(
        f"\nExperiment {context_str} accuracy on {total} questions: {accuracy:.2f}% ({correct}/{total})"
    )
    return accuracy


def main():
    # Experiment 1: Without additional context (only question and choices)
    print(
        "Running Experiment 1: Using only the question and choices (no additional context)..."
    )
    accuracy1 = run_experiment(500, use_context=False)

    # Experiment 2: With openbook.txt as context for each question
    openbook_path = os.path.join(
        "OpenBookQA-V1-Sep2018", "Data", "Main", "openbook.txt"
    )
    try:
        with open(openbook_path, "r") as f:
            context_text = f.read()
    except FileNotFoundError:
        print("openbook.txt not found!")
        context_text = ""
    print("\nRunning Experiment 2: Using openbook.txt as additional context...")
    accuracy2 = run_experiment(500, use_context=True, context_text=context_text)
    print(f"\nEXPERIMENT RESULTS:")
    print(f"Experiment 1 Accuracy: {accuracy1:.2f}%")
    print(f"Experiment 2 Accuracy: {accuracy2:.2f}%")


if __name__ == "__main__":
    main()
