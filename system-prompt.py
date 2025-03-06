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

# Define a system message with general instructions for the assistant.
SYSTEM_MESSAGE = (
    "You are a helpful assistant specialized in answering multiple-choice questions. "
    "Use any provided relevant facts to guide your answer. "
    "Carefully think about the question."
    "Respond with one of the options A, B, C, or D and include a final line formatted as 'Final Answer: <letter>'."
)


def call_deepseek(prompt, model_name, system_message=SYSTEM_MESSAGE):
    """
    Sends a prompt to the DeepSeek API using the specified model and returns the model's answer text.
    The prompt is split into a system message (general instructions) and a user message (specific question details).
    """
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
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


def process_question_for_model(
    model, model_index, question, question_index, selected_facts
):
    """
    Processes a single question for a given model.
    Constructs a user prompt that includes any selected facts, then the question and its choices.
    Returns a tuple: (model_index, question_index, predicted_int)
    """
    qid = question.get("id", "unknown")
    question_stem = question["question"]["stem"]
    choices = question["question"]["choices"]
    answer_key = question["answerKey"]

    # Build the user part of the prompt.
    user_prompt = ""
    if selected_facts:
        user_prompt += "Relevant facts:\n"
        for fact in selected_facts:
            user_prompt += f"- {fact}\n"
        user_prompt += "\n"

    user_prompt += f"Question: {question_stem}\n"
    user_prompt += "Choices:\n"
    for choice in choices:
        user_prompt += f"{choice['label']}: {choice['text']}\n"
    user_prompt += (
        "\nPlease choose the correct answer label (A, B, C, or D) and "
        "state your final decision at the end of your answer in the format 'Final Answer: <label>'."
    )

    # Call DeepSeek with both system and user messages.
    answer_text = call_deepseek(user_prompt, model)
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
    Runs the multi-model challenge for a specified number of questions.
    For each question, all 9 models are queried concurrently.
    Predictions are stored in a 2D array (rows = models, columns = questions).
    Also prints statistics:
      - Saves the 2D results array to results_matrix.json.
      - Prints individual model accuracies.
      - Prints questions with the most disagreement.
      - Prints each model’s answer for questions where the consensus is incorrect.
      - Finally, computes and prints majority vote accuracy.
    """
    # Load questions from dev.jsonl
    dev_file = os.path.join("OpenBookQA-V1-Sep2018", "Data", "Main", "dev.jsonl")
    with open(dev_file, "r") as f:
        lines = list(islice(f, questions_num))
    questions = [json.loads(line) for line in lines if line.strip()]
    num_questions = len(questions)

    # Load reranked facts from reranked_facts.json if available.
    reranked_path = "reranked_facts.json"
    facts_by_question = {}
    if os.path.exists(reranked_path):
        try:
            with open(reranked_path, "r") as f:
                facts_by_question = json.load(f)
            print(
                "\nUsing selected facts (hints) from reranked_facts.json for prompts."
            )
        except Exception as e:
            print("Error reading reranked_facts.json:", e)
    else:
        print("No reranked_facts.json file found; proceeding without additional hints.")

    # List of 9 models to use
    models = [
        "dolphin-mixtral:latest",
        "aratan/qwen2.5-14bu:latest",
        "deepseek-r1:32b",
        "aratan/DeepSeek-R1-32B-Uncensored:latest",
        "mistral:latest",
        "llama3.1:8b",
        "phi4:latest",
        "qwen2.5:32b",
        "openthinker:32b",
    ]
    num_models = len(models)

    # Initialize 2D array for predictions and expected answers
    results_matrix = [[None for _ in range(num_questions)] for _ in range(num_models)]
    expected_answers = [LETTER_TO_INT.get(q["answerKey"], None) for q in questions]

    tasks = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        for model_index, model in enumerate(models):
            for question_index, question in enumerate(questions):
                # Look up selected facts for this question id if available.
                qid = question.get("id", "unknown")
                selected_facts = facts_by_question.get(qid, {}).get(
                    "selected_facts", None
                )
                task = executor.submit(
                    process_question_for_model,
                    model,
                    model_index,
                    question,
                    question_index,
                    selected_facts,
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

    # Compute and print accuracy for each individual model.
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

    # Compute majority vote for each question.
    majority_votes = []
    for q_index in range(num_questions):
        votes = [results_matrix[m][q_index] for m in range(num_models)]
        filtered_votes = [v for v in votes if v is not None]
        if filtered_votes:
            majority = Counter(filtered_votes).most_common(1)[0][0]
        else:
            majority = None
        majority_votes.append(majority)

    # For questions where the majority vote is incorrect, print each model's answer.
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
    questions_num = 20  # Adjust the number of questions as needed.
    run_experiment(questions_num)


if __name__ == "__main__":
    main()
