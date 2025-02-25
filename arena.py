import os
import json
import re
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# Load environment variables from .env file
load_dotenv()
JWT_TOKEN = os.getenv("JWT_TOKEN")

# API endpoint (using the working endpoint from your curl example)
API_ENDPOINT = "http://localhost:3000/api/chat/completions"

# Mapping for letters to integers (A->0, B->1, C->2, D->3)
LETTER_TO_INT = {"A": 0, "B": 1, "C": 2, "D": 3}


def call_deepseek(prompt, model_name):
    """
    Sends a prompt to the DeepSeek API using the specified model and returns the model's answer text.
    """
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
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
        answer_text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print("Error parsing response:", e)
        return None
    return answer_text


def extract_final_answer(answer_text):
    """
    Extracts the final answer label (A-D) from the answer text.
    Expects a line formatted as: "Final Answer: <label>"
    """
    match = re.search(r"Final Answer:\s*([A-D])", answer_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def process_question_for_model(model, model_index, question, question_index):
    """
    Processes a single question for a given model:
      - Builds the prompt (without context).
      - Calls the API using the specified model.
      - Extracts the final answer label.
      - Prints the expected answer and full response.
    Returns a tuple: (model_index, question_index, predicted_int)
    """
    qid = question.get("id", "unknown")
    question_stem = question["question"]["stem"]
    choices = question["question"]["choices"]
    answer_key = question["answerKey"]

    # Build the prompt without context
    prompt = f"Question: {question_stem}\n"
    prompt += "Choices:\n"
    for choice in choices:
        prompt += f"{choice['label']}: {choice['text']}\n"
    prompt += (
        "\nPlease choose the correct answer label (A, B, C, or D) and "
        "state your final decision at the end of your answer in plaintext in the format 'Final Answer: <label>'. YOU MUST GIVE A FINAL ANSWER EVEN IF UNSURE."
        "\nYOU MUST PUT 'Final Answer: <label>' at the end of your answer!"
    )

    answer_text = call_deepseek(prompt, model)
    final_label = extract_final_answer(answer_text) if answer_text else None

    print(f"\n--- Question {qid} (Model: {model}) ---")
    print(f"Expected Answer: {answer_key}")
    if answer_text is None:
        print("DeepSeek Response: No answer received.")
    else:
        print("DeepSeek Response:")
        print(answer_text)
    if final_label is None:
        print("Could not extract final answer label from DeepSeek response.\n")
    else:
        print(f"Extracted Final Answer Label: {final_label}\n")

    predicted_int = LETTER_TO_INT.get(final_label, None) if final_label else None
    return model_index, question_index, predicted_int


def majority_vote(votes):
    """
    Computes the majority vote from a list of integers (0-3).
    Ignores any None values.
    Returns the majority integer vote or None if no valid votes.
    """
    filtered_votes = [v for v in votes if v is not None]
    if not filtered_votes:
        return None
    count = Counter(filtered_votes)
    majority, _ = count.most_common(1)[0]
    return majority


def run_experiment(questions_num):
    """
    Runs the challenge (without context) for a specified number of questions.
    For each question, all models are queried concurrently.
    The predictions are stored in a 2D array (rows = models, columns = questions).
    Then, extra statistics are printed:
      - The complete 2D results array is printed and saved to a JSON file.
      - Accuracy stats of each individual model.
      - Questions with the most disagreement among models.
      - For questions where the consensus (majority vote) is wrong, print each model's answer.
    Finally, the majority vote accuracy is computed and printed.
    """
    # Load questions from file
    dev_file = os.path.join("OpenBookQA-V1-Sep2018", "Data", "Main", "dev.jsonl")
    with open(dev_file, "r") as f:
        lines = list(islice(f, questions_num))
    questions = [json.loads(line) for line in lines if line.strip()]
    num_questions = len(questions)

    # List of models to use
    models = [
        "dolphin-mixtral:latest",
        "aratan/qwen2.5-14bu:latest",
        "deepseek-r1:32b",
        "aratan/DeepSeek-R1-32B-Uncensored:latest",
        "mistral:latest",
        "deepscaler:latest",
        "llama3.1:8b",
        "phi4:latest",
        "qwen2.5:32b",
        "openthinker:32b",
    ]
    num_models = len(models)

    # Initialize 2D array for predictions and store expected answers
    results_matrix = [[None for _ in range(num_questions)] for _ in range(num_models)]
    expected_answers = [LETTER_TO_INT.get(q["answerKey"], None) for q in questions]

    tasks = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        for model_index, model in enumerate(models):
            for question_index, question in enumerate(questions):
                task = executor.submit(
                    process_question_for_model,
                    model,
                    model_index,
                    question,
                    question_index,
                )
                tasks.append(task)

        for future in tqdm(
            as_completed(tasks),
            total=len(tasks),
            desc="Processing model-question pairs",
        ):
            model_index, question_index, predicted_int = future.result()
            results_matrix[model_index][question_index] = predicted_int

    # Save the 2D array and extra info to a JSON file.
    output_data = {
        "models": models,
        "results_matrix": results_matrix,
        "expected_answers": expected_answers,
    }
    with open("results_matrix.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\n2D Results Matrix (rows are models, columns are questions):")
    for model_idx, row in enumerate(results_matrix):
        print(f"{models[model_idx]}: {row}")
    print("Results matrix saved to results_matrix.json\n")

    # Compute and print accuracy for each individual model
    print("Individual Model Accuracies:")
    for model_idx, model in enumerate(models):
        row = results_matrix[model_idx]
        correct_count = sum(
            1
            for q in range(num_questions)
            if row[q] is not None and row[q] == expected_answers[q]
        )
        accuracy_model = (
            (correct_count / num_questions * 100) if num_questions > 0 else 0
        )
        print(f"{model}: {accuracy_model:.2f}% ({correct_count}/{num_questions})")
    print("")

    # Compute disagreement: for each question, count unique predictions (ignoring None)
    disagreement_stats = []
    for q_index in range(num_questions):
        votes = [
            results_matrix[m][q_index]
            for m in range(num_models)
            if results_matrix[m][q_index] is not None
        ]
        unique_votes = set(votes)
        disagreement_stats.append((q_index, len(unique_votes), unique_votes))
    # Filter questions with disagreement (more than 1 unique vote) and sort descending by unique count.
    disagreement_stats = [d for d in disagreement_stats if d[1] > 1]
    disagreement_stats.sort(key=lambda x: x[1], reverse=True)

    print("Questions with Most Disagreement Among Models:")
    for q_index, unique_count, unique_votes in disagreement_stats:
        print(
            f"Question {questions[q_index]['id']} - Unique answers count: {unique_count}"
        )
        print(f"Question: {questions[q_index]['question']['stem']}")
        print(
            f"Expected Answer: {questions[q_index]['answerKey']} (int: {expected_answers[q_index]})"
        )
        for m in range(num_models):
            vote = results_matrix[m][q_index]
            print(f"  {models[m]}: {vote}")
        print("")
    if not disagreement_stats:
        print("No disagreements found among models.\n")

    # Compute majority vote for each question
    majority_votes = []
    for q_index in range(num_questions):
        votes = [results_matrix[m][q_index] for m in range(num_models)]
        majority = majority_vote(votes)
        majority_votes.append(majority)

    # For questions where consensus (majority vote) is incorrect, print each model's answer
    print("Questions Where Consensus (Majority Vote) Was Incorrect:")
    for i in range(num_questions):
        if majority_votes[i] is not None and majority_votes[i] != expected_answers[i]:
            print(f"Question {questions[i]['id']} - {questions[i]['question']['stem']}")
            print(
                f"Expected Answer: {questions[i]['answerKey']} (int: {expected_answers[i]}), Majority Vote: {majority_votes[i]}"
            )
            for m in range(num_models):
                vote = results_matrix[m][i]
                print(f"  {models[m]}: {vote}")
            print("")
    print("")

    # Finally, compute and print majority vote accuracy.
    correct_majority = sum(
        1
        for i in range(num_questions)
        if majority_votes[i] is not None and majority_votes[i] == expected_answers[i]
    )
    majority_accuracy = (
        (correct_majority / num_questions * 100) if num_questions > 0 else 0
    )
    print(
        f"Majority vote accuracy on {num_questions} questions: {majority_accuracy:.2f}% ({correct_majority}/{num_questions})"
    )
    return results_matrix, majority_votes, majority_accuracy


def main():
    questions_num = 500  # Change this number as needed.
    run_experiment(questions_num)


if __name__ == "__main__":
    main()
