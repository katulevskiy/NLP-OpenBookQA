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
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"model": model_name, "messages": messages}
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
    match = re.search(r"Final Answer:\s*([A-D])", answer_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def majority_vote(votes):
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
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
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
    prompt = (
        f"Given a question, split it into exactly {num_subqueries} smaller questions that need to be answered "
        "in order to answer the overall question. Do not give an answer to the original question, just sub-questions that need to be answered\n\n"
        f"Question: '{question_text}'\nOutput:"
    )
    answer_text = call_deepseek_for_splitting(prompt, model=model)
    if not answer_text:
        return []
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
    if os.path.exists(subqueries_path) and os.path.getsize(subqueries_path) > 0:
        with open(subqueries_path, "r") as f:
            subqueries_data = json.load(f)
    else:
        subqueries_data = {}
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
    prompt = "We have derived the following sub-questions from the main question:\n"
    for i, sq in enumerate(subqueries, 1):
        prompt += f"Sub-question {i}: {sq}\n"
    prompt += "\n" + f"Main Question: {question_stem}\nChoices:\n"
    for choice in choices:
        prompt += f"{choice['label']}: {choice['text']}\n"
    prompt += (
        "\nPlease answer each sub-question briefly, then provide your final answer (A, B, C, or D). "
        "Include the line 'Final Answer: <letter>' at the end.\n"
    )
    return prompt


def process_question_for_model(
    model, model_index, question, question_index, subqueries, store_full_responses
):
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
    models,
    output_file="results_matrix.json",
    responses_file="all_model_responses.json",
):
    num_models = len(models)
    num_questions = len(questions)
    results_matrix = [[None for _ in range(num_questions)] for _ in range(num_models)]
    expected_answers = [LETTER_TO_INT.get(q["answerKey"], None) for q in questions]
    all_model_answers = {}
    total_tasks = num_models * num_questions
    pbar = tqdm(total=total_tasks, desc="Evaluating models")
    for model_index, model in enumerate(models):
        print(f"\nEvaluating all questions using model: {model}")
        tasks = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            for question_index, question in enumerate(questions):
                qid = question.get("id", "unknown")
                subqs = subqueries_data.get(qid, {}).get("subqueries", [])
                tasks.append(
                    executor.submit(
                        process_question_for_model,
                        model,
                        model_index,
                        question,
                        question_index,
                        subqs,
                        True,
                    )
                )
            for future in tasks:
                m_idx, q_idx, predicted_int, full_answer_text = future.result()
                results_matrix[m_idx][q_idx] = predicted_int
                qid = questions[q_idx].get("id", "unknown")
                if qid not in all_model_answers:
                    all_model_answers[qid] = {}
                all_model_answers[qid][model] = full_answer_text
                pbar.update(1)
    pbar.close()
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
    return results_matrix, expected_answers, all_model_answers


###############################################################################
#                         Step 3: Aggregator Phase                             #
###############################################################################

AGGREGATOR_SYSTEM_MESSAGE = (
    "You are a meta-reasoning system. You have the main question and the answers that various models have given. "
    "Taking the original question into account, analyze the set of answers and select the best one. "
    "Pick the single best final answer (A, B, C, or D). "
    "You must produce 'Final Answer: <letter>' at the end. "
    "Example: 'Final Answer: B'. DO NOT INCLUDE ANY STYLING OR LATEX."
)

AGGREGATOR_SYSTEM_MESSAGE_2 = (
    "You are a meta-reasoning system. You have the main question and the answers that various models have given. "
    "Taking the original question into account, analyze the set of answers and select the least bad. "
    "Pick the single best final answer (A, B, C, or D). "
    "You must produce 'Final Answer: <letter>' at the end. "
    "Example: 'Final Answer: B'. DO NOT INCLUDE ANY STYLING OR LATEX."
)


def build_aggregator_prompt(question_stem, choices, all_model_responses):
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
        "Based on the above responses, pick the best final answer (A, B, C, or D)."
    )
    return prompt


def run_aggregator_phase_single(
    questions,
    all_model_answers,
    aggregator_model="deepseek-r1:32b",
    system_message=AGGREGATOR_SYSTEM_MESSAGE,
    output_path="aggregator_answers.json",
):
    aggregator_results = {}
    tasks = {}
    max_retries = 20
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
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ]
            tasks[executor.submit(call_deepseek, messages, aggregator_model)] = (
                qid,
                messages,
            )
        for future in tqdm(
            as_completed(tasks), total=len(tasks), desc="Aggregator Phase"
        ):
            qid, messages = tasks[future]
            try:
                agg_response = future.result()
            except Exception as e:
                print(f"Aggregator error for question {qid}: {e}")
                agg_response = ""
            final_label = extract_final_answer(agg_response) if agg_response else None
            retry_count = 0
            while final_label is None and retry_count < max_retries:
                print(
                    f"Retrying aggregator for question {qid} (attempt {retry_count+1})..."
                )
                agg_response = call_deepseek(messages, aggregator_model)
                final_label = (
                    extract_final_answer(agg_response) if agg_response else None
                )
                retry_count += 1
            if final_label is None:
                print(
                    f"Aggregator still could not extract a valid answer for question {qid} after {max_retries} retries."
                )
            aggregator_results[qid] = LETTER_TO_INT.get(final_label, None)
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
                print("Aggregator Output:")
                print(agg_response)
                print(f"Extracted Aggregator Final Answer: {final_label}")
                print(f"Expected Answer: {expected}\n")
    with open(output_path, "w") as f:
        json.dump(aggregator_results, f, indent=2)
    return aggregator_results


###############################################################################
#                              Final Answer Computation                        #
###############################################################################


def compute_final_answers(questions, results_matrix, aggregator1, aggregator2):
    num_models = len(results_matrix)
    num_questions = len(questions)
    final_answers = {}
    for q_idx, question in enumerate(questions):
        qid = question.get("id", "unknown")
        votes = [
            results_matrix[m][q_idx]
            for m in range(num_models)
            if results_matrix[m][q_idx] is not None
        ]
        simple_vote = majority_vote(votes)
        agg1 = aggregator1.get(qid, None)
        agg2 = aggregator2.get(qid, None)
        votes_combined = [v for v in [agg1, agg2, simple_vote] if v is not None]
        if votes_combined:
            vote_counts = Counter(votes_combined)
            most_common = vote_counts.most_common()
            if most_common[0][1] > 1:
                final_vote = most_common[0][0]
            else:
                final_vote = agg1  # tie-breaker
        else:
            final_vote = None
        final_answers[qid] = final_vote
    with open("final_answers.json", "w") as f:
        json.dump(final_answers, f, indent=2)
    return final_answers


###############################################################################
#                              Step 4: Statistics                              #
###############################################################################


def print_statistics(
    questions,
    results_matrix,
    expected_answers,
    models,
    aggregator1,
    aggregator2,
    final_answers,
    aggregator_name="DeepSeek Aggregator",
):
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
        f"\nMajority vote (evaluation) accuracy: {majority_accuracy:.2f}% ({correct_majority}/{num_questions})"
    )

    agg1_correct = sum(
        1
        for q in questions
        if aggregator1.get(q.get("id", "unknown"))
        == LETTER_TO_INT.get(q["answerKey"], None)
    )
    agg2_correct = sum(
        1
        for q in questions
        if aggregator2.get(q.get("id", "unknown"))
        == LETTER_TO_INT.get(q["answerKey"], None)
    )
    final_correct = sum(
        1
        for q in questions
        if final_answers.get(q.get("id", "unknown"))
        == LETTER_TO_INT.get(q["answerKey"], None)
    )
    print(
        f"\nAggregator 1 accuracy: {agg1_correct/num_questions*100:.2f}% ({agg1_correct}/{num_questions})"
    )
    print(
        f"Aggregator 2 accuracy: {agg2_correct/num_questions*100:.2f}% ({agg2_correct}/{num_questions})"
    )
    print(
        f"Final combined answer accuracy: {final_correct/num_questions*100:.2f}% ({final_correct}/{num_questions})\n"
    )


###############################################################################
#                                  Main Flow                                   #
###############################################################################


def main():
    # Collect all parameters first
    params = {}
    params["questions_num"] = int(input("\nENTER NUMBER OF QUESTIONS TO PROCESS: "))
    # Step selection (default all)
    steps_input = input(
        "Enter step numbers to run (space separated, default: 1 2 3): "
    ).strip()
    if steps_input:
        try:
            params["steps"] = set(int(x) for x in steps_input.split())
        except Exception as e:
            print("Error parsing steps; defaulting to all steps.")
            params["steps"] = {1, 2, 3}
    else:
        params["steps"] = {1, 2, 3}
    # Splitting model selection
    SPLITTING_MODEL_OPTIONS = [
        "phi4:latest",
        "dolphin-mixtral:latest",
        "aratan/qwen2.5-14bu:latest",
        "deepseek-r1:32b",
        "mistral:latest",
        "qwen2.5:32b",
    ]
    print("\nAvailable Splitting Models:")
    for i, model in enumerate(SPLITTING_MODEL_OPTIONS, start=1):
        print(f"{i}. {model}")
    split_model_input = input(
        "Enter number for question splitting model (default: 1): "
    ).strip()
    try:
        params["splitting_model"] = (
            SPLITTING_MODEL_OPTIONS[int(split_model_input) - 1]
            if split_model_input
            else SPLITTING_MODEL_OPTIONS[0]
        )
    except Exception as e:
        print("Error parsing input; defaulting to first option.")
        params["splitting_model"] = SPLITTING_MODEL_OPTIONS[0]

    # Evaluation models selection
    EVAL_MODEL_OPTIONS = [
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
    print("\nAvailable Evaluation Models:")
    for i, model in enumerate(EVAL_MODEL_OPTIONS, start=1):
        print(f"{i}. {model}")
    eval_input = input(
        "Enter evaluation model numbers (space separated, default: all): "
    ).strip()
    if eval_input:
        try:
            indices = [int(x) - 1 for x in eval_input.split()]
            params["selected_eval_models"] = [
                EVAL_MODEL_OPTIONS[i]
                for i in indices
                if 0 <= i < len(EVAL_MODEL_OPTIONS)
            ]
        except Exception as e:
            print("Error parsing input; defaulting to all models.")
            params["selected_eval_models"] = EVAL_MODEL_OPTIONS
    else:
        params["selected_eval_models"] = EVAL_MODEL_OPTIONS

    # Aggregator model selection for first aggregator
    AGG_MODEL_OPTIONS = [
        "openthinker:32b",
        "deepseek-r1:32b",
        "dolphin-mixtral:latest",
        "aratan/qwen2.5-14bu:latest",
        "mistral:latest",
        "phi4:latest",
        "qwen2.5:32b",
    ]
    print("\nAvailable Aggregator Models (for first aggregator):")
    for i, model in enumerate(AGG_MODEL_OPTIONS, start=1):
        print(f"{i}. {model}")
    agg_input = input("Enter aggregator model number (default: 1): ").strip()
    try:
        params["aggregator_model_1"] = (
            AGG_MODEL_OPTIONS[int(agg_input) - 1] if agg_input else AGG_MODEL_OPTIONS[0]
        )
    except Exception as e:
        print("Error parsing input; defaulting to first option.")
        params["aggregator_model_1"] = AGG_MODEL_OPTIONS[0]

    # Aggregator model selection for second aggregator
    print("\nAvailable Aggregator Models (for second aggregator):")
    for i, model in enumerate(AGG_MODEL_OPTIONS, start=1):
        print(f"{i}. {model}")
    agg_input_2 = input(
        "Enter aggregator model number for second aggregator (default: 1): "
    ).strip()
    try:
        params["aggregator_model_2"] = (
            AGG_MODEL_OPTIONS[int(agg_input_2) - 1]
            if agg_input_2
            else AGG_MODEL_OPTIONS[0]
        )
    except Exception as e:
        print("Error parsing input; defaulting to first option.")
        params["aggregator_model_2"] = AGG_MODEL_OPTIONS[0]

    # Now all parameters are collected; print them out for confirmation.
    print("\nParameters:")
    for key, value in params.items():
        print(f"{key}: {value}")

    # 2) Load dev questions
    questions_num = params["questions_num"]
    dev_file = os.path.join("OpenBookQA-V1-Sep2018", "Data", "Main", "dev.jsonl")
    with open(dev_file, "r") as f:
        lines = list(islice(f, questions_num))
    questions = [json.loads(line) for line in lines if line.strip()]
    print(f"\nLoaded {len(questions)} questions.")

    # Step 1: Splitting
    if 1 in params["steps"]:
        num_subqueries = 3
        subqueries_data = split_questions_into_subqueries(
            questions, "subqueries.json", num_subqueries, params["splitting_model"]
        )
    else:
        if os.path.exists("subqueries.json") and os.path.getsize("subqueries.json") > 0:
            with open("subqueries.json", "r") as f:
                subqueries_data = json.load(f)
        else:
            subqueries_data = {}

    # Step 2: Evaluation
    if 2 in params["steps"]:
        results_matrix, expected_answers, all_model_answers = run_normal_evaluation(
            questions,
            subqueries_data,
            params["selected_eval_models"],
            output_file="results_matrix.json",
            responses_file="all_model_responses.json",
        )
    else:
        if (
            os.path.exists("results_matrix.json")
            and os.path.getsize("results_matrix.json") > 0
        ):
            with open("results_matrix.json", "r") as f:
                eval_data = json.load(f)
            results_matrix = eval_data["results_matrix"]
            expected_answers = eval_data["expected_answers"]
        else:
            results_matrix, expected_answers = None, None
        if (
            os.path.exists("all_model_responses.json")
            and os.path.getsize("all_model_responses.json") > 0
        ):
            with open("all_model_responses.json", "r") as f:
                all_model_answers = json.load(f)
        else:
            all_model_answers = {}

    # Step 3: Aggregation
    if 3 in params["steps"]:
        aggregator_answers1 = run_aggregator_phase_single(
            questions,
            all_model_answers,
            aggregator_model=params["aggregator_model_1"],
            system_message=AGGREGATOR_SYSTEM_MESSAGE,
            output_path="aggregator_answers1.json",
        )
        aggregator_answers2 = run_aggregator_phase_single(
            questions,
            all_model_answers,
            aggregator_model=params["aggregator_model_2"],
            system_message=AGGREGATOR_SYSTEM_MESSAGE_2,
            output_path="aggregator_answers2.json",
        )
    else:
        if (
            os.path.exists("aggregator_answers1.json")
            and os.path.getsize("aggregator_answers1.json") > 0
        ):
            with open("aggregator_answers1.json", "r") as f:
                aggregator_answers1 = json.load(f)
        else:
            aggregator_answers1 = {}
        if (
            os.path.exists("aggregator_answers2.json")
            and os.path.getsize("aggregator_answers2.json") > 0
        ):
            with open("aggregator_answers2.json", "r") as f:
                aggregator_answers2 = json.load(f)
        else:
            aggregator_answers2 = {}

    # Compute simple majority vote from evaluation results.
    simple_majority = {}
    num_eval_models = len(params["selected_eval_models"])
    for q_idx, question in enumerate(questions):
        qid = question.get("id", "unknown")
        votes = []
        for m in range(num_eval_models):
            vote = results_matrix[m][q_idx]
            if vote is not None:
                votes.append(vote)
        simple_majority[qid] = majority_vote(votes)

    # Compute final answers by taking majority vote among aggregator1, aggregator2, and simple majority vote.
    final_answers = {}
    for question in questions:
        qid = question.get("id", "unknown")
        agg1 = aggregator_answers1.get(qid, None)
        agg2 = aggregator_answers2.get(qid, None)
        simp = simple_majority.get(qid, None)
        votes_combined = [v for v in [agg1, agg2, simp] if v is not None]
        if votes_combined:
            vote_counts = Counter(votes_combined)
            most_common = vote_counts.most_common()
            if most_common[0][1] > 1:
                final_vote = most_common[0][0]
            else:
                final_vote = agg1  # tie-breaker
        else:
            final_vote = None
        final_answers[qid] = final_vote
    with open("final_answers.json", "w") as f:
        json.dump(final_answers, f, indent=2)
    print("\nFinal answers saved to final_answers.json")

    # Step 4: Print final statistics
    if results_matrix is not None and expected_answers is not None:
        agg1_correct = sum(
            1
            for q in questions
            if aggregator_answers1.get(q.get("id", "unknown"))
            == LETTER_TO_INT.get(q["answerKey"], None)
        )
        agg2_correct = sum(
            1
            for q in questions
            if aggregator_answers2.get(q.get("id", "unknown"))
            == LETTER_TO_INT.get(q["answerKey"], None)
        )
        final_correct = sum(
            1
            for q in questions
            if final_answers.get(q.get("id", "unknown"))
            == LETTER_TO_INT.get(q["answerKey"], None)
        )
        print(
            f"\nAggregator 1 accuracy: {agg1_correct/len(questions)*100:.2f}% ({agg1_correct}/{len(questions)})"
        )
        print(
            f"Aggregator 2 accuracy: {agg2_correct/len(questions)*100:.2f}% ({agg2_correct}/{len(questions)})"
        )
        print(
            f"Final combined answer accuracy: {final_correct/len(questions)*100:.2f}% ({final_correct}/{len(questions)})\n"
        )
        print_statistics(
            questions,
            results_matrix,
            expected_answers,
            params["selected_eval_models"],
            aggregator_answers1,
            aggregator_answers2,
            final_answers,
            aggregator_name="DeepSeek Aggregator",
        )


if __name__ == "__main__":
    main()
