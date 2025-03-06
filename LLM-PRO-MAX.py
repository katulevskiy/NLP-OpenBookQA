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

API_ENDPOINT = "http://localhost:3000/api/chat/completions"

LETTER_TO_INT = {"A": 0, "B": 1, "C": 2, "D": 3}
INT_TO_LETTER = {v: k for k, v in LETTER_TO_INT.items()}

###############################################################################
#                           Helper / Utility Functions                        #
###############################################################################


def call_deepseek(messages, model_name):
    """
    Sends a list of messages to the DeepSeek API using the specified model and returns the model's answer text.
    'messages' should be a list of dicts: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
    }
    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print("Error parsing response:", e)
        return None


def extract_final_answer(answer_text):
    """
    Extracts the final answer label (A-D) from the answer text.
    Expects a line formatted as: "Final Answer: <label>"
    """
    match = re.search(r"Final Answer:\s*([A-D])", answer_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def majority_vote(votes):
    """
    Computes the majority vote from a list of integers (0-3), ignoring None.
    Returns the majority integer vote or None if no valid votes.
    """
    filtered_votes = [v for v in votes if v is not None]
    if not filtered_votes:
        return None
    count = Counter(filtered_votes)
    majority, _ = count.most_common(1)[0]
    return majority


###############################################################################
#                      Step 1: Splitting Questions into Sub-Questions         #
###############################################################################


def call_deepseek_for_splitting(prompt, model="phi4:latest"):
    """
    Sends a prompt to the DeepSeek API using the specified model for splitting, and returns the model's answer text.
    """
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,  # use the provided model
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
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print("Error parsing response:", e)
        return None


def generate_subqueries(
    question_text: str, num_subqueries: int = 3, model="phi4:latest"
) -> list:
    """
    Uses the provided splitting model to generate subqueries from a given question.
    """
    prompt = (
        f"Given a question, split it into exactly {num_subqueries} smaller questions that need to be answered "
        "in order to answer the overall question. Do not give an answer to the original question, just sub-questions that need to be answered\n\n"
        f"Question: '{question_text}'\nOutput:"
    )
    answer_text = call_deepseek_for_splitting(prompt, model=model)
    if not answer_text:
        return []
    # Try to extract a numbered list
    subqueries = re.findall(r"\d+\.\s*(.*)", answer_text)
    return (
        subqueries[:num_subqueries]
        if subqueries
        else answer_text.split("\n")[:num_subqueries]
    )


def split_questions_into_subqueries(
    questions,
    subqueries_path="subqueries.json",
    num_subqueries=3,
    splitting_model="phi4:latest",
):
    """
    Processes each question from OpenBookQA. For each question:
      - Assigns a unique id if missing.
      - If an entry for that id exists in subqueries.json, skip splitting.
      - Otherwise, split the question into sub-questions using the specified splitting model.
      - This phase is parallelized with up to 50 workers.
      - The subqueries are stored in subqueries.json.
    Returns a dict: {question_id: {"subqueries": [list_of_subqueries]}}.
    """
    if os.path.exists(subqueries_path) and os.path.getsize(subqueries_path) > 0:
        with open(subqueries_path, "r") as f:
            subqueries_data = json.load(f)
    else:
        subqueries_data = {}

    new_splits = {}
    tasks = []
    for index, question in enumerate(questions):
        qid = question.get("id")
        if not qid:
            qid = f"q_{index}"
            question["id"] = qid
        if qid not in subqueries_data:
            tasks.append((qid, question["question"]["stem"]))

    print(
        f"Processing {len(tasks)} questions for sub-question splitting in parallel..."
    )

    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_qid = {
            executor.submit(
                generate_subqueries, qtext, num_subqueries, splitting_model
            ): qid
            for qid, qtext in tasks
        }
        for future in tqdm(
            as_completed(future_to_qid),
            total=len(future_to_qid),
            desc="Splitting Questions",
        ):
            qid = future_to_qid[future]
            try:
                subqs = future.result()
            except Exception as e:
                subqs = [f"Error: {str(e)}"] * num_subqueries
            print(f"\nQuestion ID: {qid}")
            for i, sq in enumerate(subqs, 1):
                print(f"  SubQ{i}: {sq}")
            subqueries_data[qid] = {"subqueries": subqs}
            new_splits[qid] = {"subqueries": subqs}

    if new_splits:
        with open(subqueries_path, "w") as f:
            json.dump(subqueries_data, f, indent=2)

    return subqueries_data


###############################################################################
#                  Step 2: Normal Evaluation with Sub-Questions               #
###############################################################################

SYSTEM_MESSAGE_ANSWER = (
    "You are a helpful assistant specialized in answering multiple-choice questions. "
    "Please carefully consider any provided sub-questions (and answer them) before providing the final choice. "
    "Evaluate how answering the sub-questions is related to the initial question. Reason for your choice of answer to the initial question. "
    "Only after carefully thinking about all options, provide your final choice. "
    "Your final choice must be one of A, B, C, or D, and you must include a line at the end: 'Final Answer: <letter>'."
)


def build_main_prompt(question_stem, choices, subqueries):
    """
    Builds the user prompt to show sub-questions plus the main question and answer choices.
    """
    prompt = "We have derived the following sub-questions from the main question:\n"
    for i, sq in enumerate(subqueries, 1):
        prompt += f"Sub-question {i}: {sq}\n"
    prompt += "\n"
    prompt += f"Main Question: {question_stem}\nChoices:\n"
    for choice in choices:
        prompt += f"{choice['label']}: {choice['text']}\n"
    prompt += (
        "\nPlease answer each sub-question briefly, then provide your final answer (A, B, C, or D). "
        "Include the line 'Final Answer: <label>' at the end.\n"
    )
    return prompt


def process_question_for_model(
    model, model_index, question, question_index, subqueries, store_full_responses
):
    """
    Processes a single question for a given model. Uses the sub-questions (if any).
    Returns (model_index, question_index, predicted_int, full_answer_text).
    """
    qid = question.get("id", "unknown")
    question_stem = question["question"]["stem"]
    choices = question["question"]["choices"]
    answer_key = question["answerKey"]

    user_prompt = build_main_prompt(question_stem, choices, subqueries)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE_ANSWER},
        {"role": "user", "content": user_prompt},
    ]
    answer_text = call_deepseek(messages, model)

    final_label = extract_final_answer(answer_text) if answer_text else None
    predicted_int = LETTER_TO_INT.get(final_label, None) if final_label else None

    print(f"\n--- Question {qid} (Model: {model}) ---")
    if answer_text is None:
        print("DeepSeek Response: No answer received.")
    else:
        print("DeepSeek Response:")
        print(answer_text)
    if final_label is None:
        print("Could not extract final answer label.")
    else:
        print(f"Extracted Final Answer Label: {final_label}")
    print(f"Expected Answer: {answer_key}\n")

    return model_index, question_index, predicted_int, (answer_text or "")


def run_normal_evaluation(
    questions,
    subqueries_data,
    output_file="results_matrix.json",
    responses_file="all_responses.json",
):
    """
    Runs the multi-model challenge. Uses sub-questions in subqueries_data if present.
    Returns (results_matrix, expected_answers, all_model_answers).
    all_model_answers is a dict of the form:
        { question_id: { model_name: "full response text", ... }, ... }
    """
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
    num_questions = len(questions)

    results_matrix = [[None for _ in range(num_questions)] for _ in range(num_models)]
    expected_answers = [LETTER_TO_INT.get(q["answerKey"], None) for q in questions]
    all_model_answers = {}

    tasks = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        for model_index, model in enumerate(models):
            for question_index, question in enumerate(questions):
                qid = question.get("id", "unknown")
                subqs = subqueries_data.get(qid, {}).get("subqueries", [])
                task = executor.submit(
                    process_question_for_model,
                    model,
                    model_index,
                    question,
                    question_index,
                    subqs,
                    store_full_responses=True,
                )
                tasks.append(task)

        for future in tqdm(
            as_completed(tasks), total=len(tasks), desc="Evaluating models"
        ):
            model_index, question_index, predicted_int, full_answer_text = (
                future.result()
            )
            results_matrix[model_index][question_index] = predicted_int

            qid = questions[question_index].get("id", "unknown")
            model_name = models[model_index]
            if qid not in all_model_answers:
                all_model_answers[qid] = {}
            all_model_answers[qid][model_name] = full_answer_text

    with open(output_file, "w") as f:
        json.dump(
            {
                "models": models,
                "results_matrix": results_matrix,
                "expected_answers": expected_answers,
            },
            f,
            indent=2,
        )
    with open(responses_file, "w") as f:
        json.dump(all_model_answers, f, indent=2)

    print(f"\nResults matrix saved to {output_file}")
    print(f"Full model responses saved to {responses_file}\n")

    return results_matrix, expected_answers, models, all_model_answers


###############################################################################
#                         Step 3: Aggregator Phase                             #
###############################################################################

AGGREGATOR_SYSTEM_MESSAGE = (
    "You are a meta-reasoning system. You have the main question and the answers that various models have given. "
    "Use their responses to pick the single best final answer (A, B, C, or D). "
    "You must produce 'Final Answer: <letter>' at the end."
)


def build_aggregator_prompt(question_stem, choices, all_model_responses):
    """
    Creates a prompt that includes the question, the answer options, and the provided model responses.
    """
    prompt = "Main Question:\n"
    prompt += f"{question_stem}\n\n"
    prompt += "Choices:\n"
    for choice in choices:
        prompt += f"{choice['label']}: {choice['text']}\n"
    prompt += "\n"
    prompt += "Here are the responses from various models:\n"
    for model_name, response_text in all_model_responses.items():
        prompt += f"[{model_name}] => {response_text}\n\n"
    prompt += (
        "Based on the above responses, pick the best final answer (A, B, C, or D). "
        "Do not justify yourself at length—just give the final result. "
        "Final Answer: "
    )
    return prompt


def run_aggregator_phase(
    questions,
    all_model_answers,
    aggregator_model="deepseek-r1:32b",
    output_path="aggregator_answers.json",
):
    """
    For each question, call the aggregator model in parallel (up to 50 workers) to pick a final answer
    given all other models' responses. Returns a dict {question_id: aggregator_answer_index}.
    Also saves aggregator final choices to aggregator_answers.json.
    Additionally, prints to console the question, choices, all model responses, the aggregator output,
    the extracted final answer, and the expected answer.
    """
    aggregator_answers = {}
    tasks = {}
    with ThreadPoolExecutor(max_workers=50) as executor:
        for question in questions:
            qid = question.get("id", "unknown")
            question_stem = question["question"]["stem"]
            choices = question["question"]["choices"]
            model_responses = all_model_answers.get(qid, {})
            user_prompt = build_aggregator_prompt(
                question_stem, choices, model_responses
            )
            messages = [
                {"role": "system", "content": AGGREGATOR_SYSTEM_MESSAGE},
                {"role": "user", "content": user_prompt},
            ]
            tasks[executor.submit(call_deepseek, messages, aggregator_model)] = qid

        for future in tqdm(
            as_completed(tasks), total=len(tasks), desc="Aggregator Phase"
        ):
            qid = tasks[future]
            try:
                agg_response = future.result()
            except Exception as e:
                print(f"Aggregator error for question {qid}: {e}")
                agg_response = ""
            final_label = extract_final_answer(agg_response) if agg_response else None
            aggregator_answers[qid] = LETTER_TO_INT.get(final_label, None)

            # Print detailed aggregation info
            question = next((q for q in questions if q.get("id") == qid), None)
            if question:
                question_stem = question["question"]["stem"]
                choices = question["question"]["choices"]
                expected = question["answerKey"]
                print(f"\nAggregator Evaluation for Question ID: {qid}")
                print(f"Question: {question_stem}")
                print("Choices:")
                for choice in choices:
                    print(f"  {choice['label']}: {choice['text']}")
                print("Model responses:")
                model_responses = all_model_answers.get(qid, {})
                for m, resp in model_responses.items():
                    print(f"  {m}: {resp}")
                print("Aggregator Output:")
                print(agg_response)
                print(f"Extracted Aggregator Final Answer: {final_label}")
                print(f"Expected Answer: {expected}\n")

    with open(output_path, "w") as f:
        json.dump(aggregator_answers, f, indent=2)

    return aggregator_answers


###############################################################################
#                              Step 4: Statistics                              #
###############################################################################


def print_statistics(
    questions,
    results_matrix,
    expected_answers,
    models,
    aggregator_answers,
    aggregator_name="DeepSeek Aggregator",
):
    """
    Print the usual stats:
      - Individual model accuracy
      - Questions with the most disagreement
      - Majority vote accuracy
      - Aggregator accuracy
    """
    num_models = len(models)
    num_questions = len(questions)

    print("\nIndividual Model Accuracies:")
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

    print("\nQuestions with Most Disagreement Among Models:")
    if not disagreement_stats:
        print("No disagreements found among models.")
    else:
        for q_index, unique_count, unique_votes in disagreement_stats:
            question_id = questions[q_index]["id"]
            print(f"Question {question_id} - Unique answers count: {unique_count}")
            print(f"Question: {questions[q_index]['question']['stem']}")
            print(
                f"Expected Answer: {questions[q_index]['answerKey']} (int: {expected_answers[q_index]})"
            )
            for m in range(num_models):
                vote = results_matrix[m][q_index]
                print(f"  {models[m]}: {vote}")
            print("")

    majority_votes = []
    for q_index in range(num_questions):
        votes = [results_matrix[m][q_index] for m in range(num_models)]
        filtered_votes = [v for v in votes if v is not None]
        if filtered_votes:
            majority = Counter(filtered_votes).most_common(1)[0][0]
        else:
            majority = None
        majority_votes.append(majority)

    correct_majority = sum(
        1
        for i in range(num_questions)
        if majority_votes[i] is not None and majority_votes[i] == expected_answers[i]
    )
    majority_accuracy = (
        (correct_majority / num_questions * 100) if num_questions > 0 else 0
    )
    print(
        f"\nMajority vote accuracy: {majority_accuracy:.2f}% ({correct_majority}/{num_questions})"
    )

    aggregator_correct = 0
    for i, question in enumerate(questions):
        qid = question.get("id", "unknown")
        agg_vote = aggregator_answers.get(qid, None)
        if agg_vote == expected_answers[i]:
            aggregator_correct += 1
    aggregator_accuracy = (
        (aggregator_correct / num_questions * 100) if num_questions > 0 else 0
    )
    print(
        f"{aggregator_name} accuracy: {aggregator_accuracy:.2f}% ({aggregator_correct}/{num_questions})"
    )


###############################################################################
#                                  Main Flow                                   #
###############################################################################


def main():
    # 1) Load the dev questions
    questions_num = 3  # or however many you want
    dev_file = os.path.join("OpenBookQA-V1-Sep2018", "Data", "Main", "dev.jsonl")
    with open(dev_file, "r") as f:
        lines = list(islice(f, questions_num))
    questions = [json.loads(line) for line in lines if line.strip()]
    print(f"Loaded {len(questions)} questions.")

    # Let user choose the splitting model.
    splitting_model = (
        input("Enter model for question splitting (default: phi4:latest): ").strip()
        or "phi4:latest"
    )

    # 2) Split questions into sub-questions if not already done (parallelized).
    num_subqueries = 3
    subqueries_data = split_questions_into_subqueries(
        questions, "subqueries.json", num_subqueries, splitting_model
    )

    # 3) Normal evaluation with sub-questions
    results_matrix, expected_answers, models, all_model_answers = run_normal_evaluation(
        questions,
        subqueries_data,
        output_file="results_matrix.json",
        responses_file="all_responses.json",
    )

    # 4) Aggregator phase (parallelized)
    aggregator_answers = run_aggregator_phase(
        questions,
        all_model_answers,
        aggregator_model="deepseek-r1:32b",
        output_path="aggregator_answers.json",
    )

    # 5) Print final statistics
    print_statistics(
        questions,
        results_matrix,
        expected_answers,
        models,
        aggregator_answers,
        aggregator_name="DeepSeek Aggregator",
    )


if __name__ == "__main__":
    main()
