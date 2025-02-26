import os
import json
from collections import Counter

# Mapping between integer predictions and letters
INT_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_results(filename):
    """Load models, results_matrix, and expected_answers from results_matrix.json."""
    with open(filename, "r") as f:
        data = json.load(f)
    return data["models"], data["results_matrix"], data["expected_answers"]


def load_questions(dev_filename, count):
    """Load the first 'count' questions from dev.jsonl."""
    questions = []
    with open(dev_filename, "r") as f:
        for i, line in enumerate(f):
            if i >= count:
                break
            if line.strip():
                questions.append(json.loads(line))
    return questions


def compute_model_accuracies(results_matrix, expected_answers):
    """Compute accuracy for each model (row) based on expected answers."""
    num_questions = len(expected_answers)
    model_accuracies = []
    for row in results_matrix:
        correct = sum(
            1
            for pred, exp in zip(row, expected_answers)
            if pred is not None and pred == exp
        )
        acc = correct / num_questions * 100 if num_questions > 0 else 0
        model_accuracies.append(acc)
    return model_accuracies


def majority_vote(votes):
    """Given a list of votes (ints) ignoring None, return the majority vote.
    Also return the frequency and total count of valid votes."""
    valid_votes = [v for v in votes if v is not None]
    if not valid_votes:
        return None, 0, 0
    count = Counter(valid_votes)
    majority, freq = count.most_common(1)[0]
    return majority, freq, len(valid_votes)


def compute_consensus(results_matrix, expected_answers):
    """For each question, compute the majority vote and overall consensus accuracy."""
    num_questions = len(expected_answers)
    majority_votes = []
    consensus_correct = 0
    for j in range(num_questions):
        votes = [results_matrix[i][j] for i in range(len(results_matrix))]
        majority, _, _ = majority_vote(votes)
        majority_votes.append(majority)
        if majority is not None and majority == expected_answers[j]:
            consensus_correct += 1
    consensus_accuracy = (
        consensus_correct / num_questions * 100 if num_questions > 0 else 0
    )
    return majority_votes, consensus_accuracy


def main():
    # File paths (adjust if needed)
    results_file = "results_matrix.json"
    dev_file = os.path.join("OpenBookQA-V1-Sep2018", "Data", "Main", "dev.jsonl")

    # Load data
    models, results_matrix, expected_answers = load_results(results_file)
    num_questions = len(expected_answers)
    questions = load_questions(dev_file, num_questions)

    # Compute consensus (majority vote) for each question and consensus accuracy
    majority_votes, consensus_accuracy = compute_consensus(
        results_matrix, expected_answers
    )

    # Print questions with the most disagreement among models.
    # Here we define disagreement as the number of unique predictions (ignoring None)
    print("Questions with Most Disagreement Among Models:")
    disagreement_list = []
    for j in range(num_questions):
        votes = [
            results_matrix[i][j]
            for i in range(len(results_matrix))
            if results_matrix[i][j] is not None
        ]
        unique_votes = set(votes)
        disagreement_list.append((j, len(unique_votes), unique_votes))
    # Filter questions with disagreement > 1 and sort descending by unique vote count.
    disagreement_sorted = sorted(
        [d for d in disagreement_list if d[1] > 1], key=lambda x: x[1], reverse=True
    )
    if disagreement_sorted:
        for j, dis_count, unique_votes in disagreement_sorted:
            qid = questions[j].get("id", "unknown")
            votes_letters = [
                INT_TO_LETTER[v] for v in unique_votes if v in INT_TO_LETTER
            ]
            print(f"  Question {qid}: {dis_count} unique answers: {votes_letters}")
    else:
        print("  No significant disagreement among models.")
    print()

    # For questions where the consensus (majority vote) is incorrect, print each model's answer.
    print("Questions Where Consensus (Majority Vote) Was Incorrect:")
    for j in range(num_questions):
        consensus = majority_votes[j]
        expected = expected_answers[j]
        if consensus is None or consensus == expected:
            continue
        question = questions[j]
        qid = question.get("id", "unknown")
        print(f"Question {qid}: {question['question']['stem']}")
        print(
            f"  Expected: {INT_TO_LETTER[expected]}, Consensus: {INT_TO_LETTER.get(consensus, 'None')}"
        )
        for i, model in enumerate(models):
            pred = results_matrix[i][j]
            pred_letter = INT_TO_LETTER.get(pred, "None")
            print(f"    {model}: {pred_letter}")
        print()

    # Compute and print individual model accuracies
    model_accuracies = compute_model_accuracies(results_matrix, expected_answers)
    print("Individual Model Accuracies:")
    for model, acc in zip(models, model_accuracies):
        print(f"  {model}: {acc:.2f}%")
    print()

    # Finally, print consensus accuracy.
    correct_consensus = sum(
        1 for j in range(num_questions) if majority_votes[j] == expected_answers[j]
    )
    print(
        f"Consensus (Majority Vote) Accuracy: {consensus_accuracy:.2f}% ({correct_consensus}/{num_questions})"
    )


if __name__ == "__main__":
    main()
