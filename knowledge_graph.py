from itertools import islice
import json
import os
import requests
from tqdm import tqdm


USEFUL_EDGE_TYPES = ['/r/HasA', '/r/IsA', '/r/FormOf', '/r/UsedFor', '/r/Causes', '/r/CreatedBy', '/r/MadeOf']

def prettify_edge_label(label):
    """
    Prettifies an edge label, e.g. turning "IsA" into "is a"
    """
    result = ''
    for c in label:
        if c.isupper() and result:
            result += ' '
        result += c.lower()
    return result


def find_relevant_knowledge(sentence, num_knowledge=3):
    """
    Finds relevant knowledge from the KG for a given sentence. Returns up to num_knowledge facts.
    """
    words = [word.lower() for word in sentence.split(' ')]
    long_enough_words = [word.strip(',') for word in words if len(word) >= 4]
    words_set = set(long_enough_words)
    for word in words_set:
        # print('word:', word)
        # print('checking', current_words_set)
        response = requests.get(f'http://api.conceptnet.io/c/en/{word}?limit=100').json()
        candidates = []
        for edge in response['edges']:
            if edge['rel']['@id'] in USEFUL_EDGE_TYPES:
                # print('candidates:', edge['start']['label'], edge['end']['label'])
                candidates.append(f"{edge['start']['label'].lower()} {prettify_edge_label(edge['rel']['label'])} {edge['end']['label'].lower()}")
                if len(candidates) >= num_knowledge:
                    return candidates[:num_knowledge]

                # if edge['start']['label'] == word:
                #     print('match between', word, edge['end']['label'], edge['rel']['@id'])
                #     candidates.append()
                # elif :
                #     print('match between', word, edge['start']['label'], edge['rel']['@id'])

    return candidates[:num_knowledge]


def run_experiment(questions_num, use_context=False, context_text=""):
    """
    Processes the first several questions, extracting relevant knowledge from the KG for each one.
    """
    correct = 0
    total = 0
    dev_file = os.path.join("OpenBookQA-V1-Sep2018", "Data", "Main", "dev.jsonl")

    with open(dev_file, "r") as f:
        # Process only the first 10 questions
        lines = list(islice(f, questions_num))

    for line in tqdm(lines, total=questions_num, desc="Processing questions"):
        if not line.strip():
            continue
        data = json.loads(line)
        qid = data.get("id", "unknown")
        question_stem = data["question"]["stem"]
        choices = data["question"]["choices"]
        answer_key = data["answerKey"]

        relevant_knowledge = find_relevant_knowledge(question_stem)
        print('Relevant knowledge:', relevant_knowledge)


def main():
    # Experiment 1: Without additional context (only question and choices)
    print(
        "Running Experiment 1: Using only the question and choices (no additional context)..."
    )
    run_experiment(20, use_context=False)



if __name__ == "__main__":
    main()